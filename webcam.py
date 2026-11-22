"""Browser-camera processing for deployed apps (streamlit-webrtc).

The camera is opened by the visitor's *browser* and streamed to the app, so it
works on Streamlit Community Cloud, Docker or any server.

Design (this is what removes the lag)
-------------------------------------
* recv()   runs for every video frame and does almost nothing: it hands the
           newest frame to a worker and draws the latest known boxes on it,
           so the live picture is never delayed by the detector.
* worker   a background thread that always takes only the NEWEST frame,
           shrinks it, runs person detection, and publishes the boxes.
           Frames that arrive while it is busy are skipped, so a slow server
           gives fewer box updates instead of a growing delay.
"""
import threading
import time
from collections import deque

import av
import cv2

import heatmap as hm
from config import LEVEL_COLORS, classify

_DET_LOCK = threading.Lock()   # the shared detector is not thread-safe


class LiveProcessor:
    def __init__(self, detector, max_side=640):
        self.det = detector
        self.max_side = max_side
        self.every, self.factor, self.conf = 3, 1.0, 0.35   # `every` kept for compatibility, unused
        self.capacity, self.th = 100, (0.25, 0.50, 0.75)
        self.smooth, self.frames, self.det_fps, self.error = 0, 0, 0.0, None
        self._boxes, self._hist = [], deque(maxlen=5)
        self._latest, self._lock = None, threading.Lock()
        self._last_frame = time.time()
        self._stop = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- background detection -------------------------------------------
    def _worker(self):
        while not self._stop.is_set():
            if time.time() - self._last_frame > 15:      # stream closed -> end the thread
                break
            with self._lock:
                img, self._latest = self._latest, None
            if img is None:
                time.sleep(0.02)
                continue
            t0 = time.time()
            h, w = img.shape[:2]
            scale = min(1.0, self.max_side / max(h, w))
            small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
            try:
                with _DET_LOCK:
                    self.det.conf = self.conf
                    found = self.det.detect(small)
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.5)
                continue
            self._boxes = [(x1 / scale, y1 / scale, x2 / scale, y2 / scale, c) for x1, y1, x2, y2, c in found]
            self._hist.append(len(found) * self.factor)
            self.smooth = int(round(sorted(self._hist)[len(self._hist) // 2]))   # median: steady, no flicker
            self.det_fps = 1.0 / max(time.time() - t0, 1e-3)

    def on_ended(self):
        self._stop.set()

    # ---- per-frame (fast) -------------------------------------------------
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self._last_frame = time.time()
        self.frames += 1
        with self._lock:
            self._latest = img.copy()
        out = hm.draw_boxes(img, list(self._boxes))
        lvl = classify(self.smooth, self.capacity, self.th)
        hexc = LEVEL_COLORS[lvl]
        bgr = (int(hexc[5:7], 16), int(hexc[3:5], 16), int(hexc[1:3], 16))
        text = f"{self.smooth} people - {lvl}"
        cv2.putText(out, text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5)
        cv2.putText(out, text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2)
        return av.VideoFrame.from_ndarray(out, format="bgr24")
