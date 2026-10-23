"""Person detection for images and video.

Backends
--------
yolo : Ultralytics YOLOv8 (best accuracy; needs `pip install ultralytics`,
       downloads yolov8n.pt on first use).
hog  : OpenCV HOG+SVM pedestrian detector (no extra install, works offline,
       less accurate in dense crowds).
auto : try YOLO, silently fall back to HOG.
"""
import cv2
import numpy as np


class PersonDetector:
    def __init__(self, backend="auto", conf=0.35, imgsz=1280):
        self.conf = conf
        self.imgsz = imgsz
        self.model = None
        self.hog = None
        self.error = None
        self.backend = "hog"
        if backend in ("auto", "yolo"):
            try:
                from ultralytics import YOLO

                self.model = YOLO("yolov8n.pt")
                self.backend = "yolo"
            except Exception as exc:  # not installed / no weights / no network
                self.error = str(exc)
                if backend == "yolo":
                    raise
        if self.model is None:
            if not hasattr(cv2, "HOGDescriptor"):
                raise RuntimeError(
                    f"This OpenCV build ({cv2.__version__}) has no HOG people detector (OpenCV 5 removed it). "
                    "Fix: python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python "
                    "&& python -m pip install \"opencv-python<5\"   (or install YOLO: python -m pip install ultralytics)")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # ------------------------------------------------------------------
    def detect(self, bgr):
        """Return list of (x1, y1, x2, y2, confidence) in image coordinates."""
        if self.model is not None:
            res = self.model.predict(bgr, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            return [(*map(float, b), float(c)) for b, c in zip(xyxy, conf)]
        return self._detect_hog(bgr)

    def _detect_hog(self, bgr):
        h, w = bgr.shape[:2]
        target_w = 960 if w > 960 else (800 if w < 640 else w)
        scale = target_w / w
        img = cv2.resize(bgr, (target_w, int(h * scale)))
        rects, weights = self.hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.04)
        if len(rects) == 0:
            return []
        weights = np.asarray(weights).reshape(-1)
        keep = weights >= self.conf
        rects, weights = rects[keep], weights[keep]
        if len(rects) == 0:
            return []
        idx = cv2.dnn.NMSBoxes([list(map(int, r)) for r in rects], weights.tolist(), float(self.conf), 0.4)
        out = []
        for i in np.array(idx).reshape(-1):
            x, y, bw, bh = rects[i]
            out.append((x / scale, y / scale, (x + bw) / scale, (y + bh) / scale, float(weights[i])))
        return out


def detect_video(path, detector, every_sec=2.0, max_frames=30, progress=None):
    """Sample frames from a video. Returns (per_frame list, peak_frame, boxes_per_frame)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(fps * every_sec)), 1)
    stats, all_boxes, peak = [], [], (-1, None, None)
    i, grabbed = 0, 0
    while grabbed < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detector.detect(frame)
        stats.append({"frame": i, "time_s": round(i / fps, 1), "count": len(boxes)})
        all_boxes.append(boxes)
        if len(boxes) > peak[0]:
            peak = (len(boxes), frame.copy(), boxes)
        grabbed += 1
        i += step
        if progress:
            progress(min(grabbed / max_frames, (i / total) if total else 1.0))
    cap.release()
    if not stats:
        raise ValueError("No frames could be read from the video.")
    return stats, peak[1], all_boxes
