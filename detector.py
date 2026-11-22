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


def dedupe(boxes, iou_thr=0.45, contain_thr=0.8, min_side=12):
    """Remove duplicate detections of the same person.

    A box is dropped when a higher-confidence box already covers the same
    person: either they overlap a lot (IoU) or the smaller box lies almost
    entirely inside the bigger one (e.g. head+shoulders box inside a full-body box).
    boxes: list of (x1, y1, x2, y2, conf)
    """
    boxes = [b for b in boxes if (b[2] - b[0]) >= min_side and (b[3] - b[1]) >= min_side]
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for b in boxes:
        ab = (b[2] - b[0]) * (b[3] - b[1])
        dup = False
        for k in kept:
            iw = min(b[2], k[2]) - max(b[0], k[0])
            ih = min(b[3], k[3]) - max(b[1], k[1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            ak = (k[2] - k[0]) * (k[3] - k[1])
            if inter / (ab + ak - inter) > iou_thr or inter / min(ab, ak) > contain_thr:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


class PersonDetector:
    def __init__(self, backend="auto", conf=0.30, imgsz=1280, model_name="yolov8n.pt"):
        self.tiles = False   # True = also scan 4 magnified tiles (finds small / far people, ~5x slower)
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self.model = None
        self.hog = None
        self.hog_width = None   # None = automatic; a smaller number = faster HOG
        self.error = None
        self.backend = "hog"
        if backend in ("auto", "yolo"):
            try:
                from ultralytics import YOLO

                try:                      # leave one CPU core free so live video keeps flowing
                    import os
                    import torch

                    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
                except Exception:
                    pass
                try:
                    self.model = YOLO(model_name)
                except Exception:         # e.g. yolov8s.pt could not be downloaded -> use the bundled nano model
                    self.model_name = "yolov8n.pt"
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
        found = self._detect_once(bgr)
        if self.tiles:
            h, w = bgr.shape[:2]
            tw, th = int(w * 0.6), int(h * 0.6)
            for fx in (0, 1):
                for fy in (0, 1):
                    x0, y0 = int(fx * (w - tw)), int(fy * (h - th))
                    for x1, y1, x2, y2, c in self._detect_once(bgr[y0:y0 + th, x0:x0 + tw]):
                        found.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, c))
            found = dedupe(found)
        return found

    def _detect_once(self, bgr):
        if self.model is not None:
            res = self.model.predict(bgr, classes=[0], conf=self.conf, iou=0.5, agnostic_nms=True,
                                     imgsz=self.imgsz, verbose=False)[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            return dedupe([(*map(float, b), float(c)) for b, c in zip(xyxy, conf)])
        return self._detect_hog(bgr)

    def _detect_hog(self, bgr):
        h, w = bgr.shape[:2]
        target_w = self.hog_width or (960 if w > 960 else (800 if w < 640 else w))
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
        return dedupe(out)


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
