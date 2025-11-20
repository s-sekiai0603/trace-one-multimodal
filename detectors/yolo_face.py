# -*- coding: utf-8 -*-
"""
YOLOv8 ベースの顔検出（BBOX + score）
- 入力: BGR画像 (H,W,3) np.uint8
- 出力: boxes (N,4: x1,y1,x2,y2), scores (N,)
"""
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import cv2
import logging

log = logging.getLogger("app.yolo_face")

# デバイスやデフォルト値は config から（無ければフォールバック）
try:
    from ..config import (
        TORCH_DEVICE,
        YOLO_FACE_MODEL_PATH, YOLO_FACE_CONF, YOLO_FACE_IOU, YOLO_FACE_IMGSZ, YOLO_FACE_MAX_DET
    )
except Exception:
    TORCH_DEVICE = "cpu"
    YOLO_FACE_MODEL_PATH = "models/yolov8s-face-lindevs.pt"
    YOLO_FACE_CONF = 0.25
    YOLO_FACE_IOU = 0.45
    YOLO_FACE_IMGSZ = 640
    YOLO_FACE_MAX_DET = 200

class YOLOFaceDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf: float = YOLO_FACE_CONF,
        iou: float = YOLO_FACE_IOU,
        imgsz: int = YOLO_FACE_IMGSZ,
        max_det: int = YOLO_FACE_MAX_DET,
    ):
        self.model_path = model_path or YOLO_FACE_MODEL_PATH
        self.device = device or TORCH_DEVICE
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError("ultralytics が見つかりません。`pip install ultralytics` を実行してください。") from e

        self.model = YOLO(self.model_path)
        try: self.model.fuse()
        except Exception: pass
        try: self.model.to(self.device)
        except Exception: pass

        log.info("[YOLO-FACE] model loaded: %s (device=%s, conf=%.2f, iou=%.2f, imgsz=%s, max_det=%d)",
                 self.model_path, self.device, self.conf, self.iou, str(self.imgsz), self.max_det)

        self._dbg_cnt = 0

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        顔検出のみ（描画なし）
        Returns:
            boxes_xyxy: (N,4) float32 [x1,y1,x2,y2]
            scores:     (N,)  float32
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

        H, W = frame_bgr.shape[:2]
        self._dbg_cnt += 1
        if self._dbg_cnt <= 3:
            log.info("[YOLO-FACE] infer: HxW=%dx%d, conf=%.2f, iou=%.2f, imgsz=%s, max_det=%d",
                     H, W, self.conf, self.iou, str(self.imgsz), self.max_det)

        # Ultralytics の推論（BGR→RGB）
        results = self.model.predict(
            source=frame_bgr[..., ::-1],
            imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            max_det=self.max_det, verbose=False, device=self.device
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            log.debug("[YOLO-FACE] dets=0")
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

        b = results[0].boxes
        xyxy = b.xyxy.detach().cpu().numpy().astype(np.float32)  # (N,4)
        score = b.conf.detach().cpu().numpy().astype(np.float32) # (N,)
        log.debug("[YOLO-FACE] dets=%d", xyxy.shape[0])
        return xyxy, score

    @staticmethod
    def draw(
        frame_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        scores: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        show_score: bool = True,
    ) -> None:
        if boxes_xyxy is None or boxes_xyxy.size == 0:
            return
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        t = int(thickness)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
            if show_score and scores is not None and i < len(scores):
                sc = float(scores[i])
                label = f"{sc:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y0 = max(0, y1 - 3)
                cv2.rectangle(frame_bgr, (x1, y0 - th - baseline), (x1 + tw + 2, y0), color, -1)
                cv2.putText(frame_bgr, label, (x1 + 1, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    def detect_and_draw(
        self,
        frame_bgr: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        show_score: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        検出してそのまま描画まで行うユーティリティ
        """
        boxes, scores = self.detect(frame_bgr)
        YOLOFaceDetector.draw(frame_bgr, boxes, scores, color=color, thickness=thickness, show_score=show_score)
        return boxes, scores
