# -*- coding: utf-8 -*-
"""
ada_face_utils.py — AdaFace用のズーム＆BBOX平滑化ユーティリティ
- inflate_boxes_xyxy: BBOXをmargin[%]だけ外側に広げて台形崩れを防ぎつつクリップ
- smooth_boxes_ema  : フレーム間のBBOXを指数移動平均(EMA)で平滑化
- choose_boxes      : 平滑化結果があればそれを、無ければ元箱を返す簡易切替
"""
from __future__ import annotations
from typing import Optional
import numpy as np

def _clamp_boxes_xyxy(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """ [x1,y1,x2,y2] を画面内にクリップ """
    b = boxes.astype(np.float32, copy=True)
    b[:, 0] = np.clip(b[:, 0], 0, max(0, w - 1))
    b[:, 1] = np.clip(b[:, 1], 0, max(0, h - 1))
    b[:, 2] = np.clip(b[:, 2], 1, max(1, w))
    b[:, 3] = np.clip(b[:, 3], 1, max(1, h))
    # 最低1pxは確保
    b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1)
    b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1)
    return b

def inflate_boxes_xyxy(boxes: np.ndarray, frame_shape_hw, margin: float = 0.30) -> np.ndarray:
    """
    BBOXを margin 割合だけ全方向に広げる（顔が小さい時の“先ズーム”）
    - boxes: (N,4) xyxy
    - frame_shape_hw: (H, W)
    """
    if boxes is None or len(boxes) == 0:
        return boxes
    H, W = int(frame_shape_hw[0]), int(frame_shape_hw[1])
    b = boxes.astype(np.float32, copy=True)
    w = (b[:, 2] - b[:, 0])
    h = (b[:, 3] - b[:, 1])
    dx = w * float(margin)
    dy = h * float(margin)
    b[:, 0] = b[:, 0] - dx
    b[:, 1] = b[:, 1] - dy
    b[:, 2] = b[:, 2] + dx
    b[:, 3] = b[:, 3] + dy
    return _clamp_boxes_xyxy(b, H, W)

def smooth_boxes_ema(curr_boxes: Optional[np.ndarray],
                     prev_boxes: Optional[np.ndarray],
                     alpha: float = 0.5) -> Optional[np.ndarray]:
    """
    BBOXの指数移動平均（指数平滑）。同じ順序の箱対応を仮定（検出の順序が大きく入れ替わらない前提）。
    - curr_boxes: 今フレームの (N,4)
    - prev_boxes: 直近EMA結果 (N,4) か None
    - alpha     : 0..1（大きいほど現在値寄り）
    """
    if curr_boxes is None or len(curr_boxes) == 0:
        return curr_boxes
    c = curr_boxes.astype(np.float32)
    if prev_boxes is None or len(prev_boxes) != len(c):
        return c
    return (alpha * c + (1.0 - alpha) * prev_boxes.astype(np.float32)).astype(np.float32)

def choose_boxes(smooth_boxes: Optional[np.ndarray],
                 raw_boxes: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """ 平滑化があればそれを、無ければ生を返す """
    if smooth_boxes is not None and len(smooth_boxes) > 0:
        return smooth_boxes
    return raw_boxes
