# src/utils/osnet_utils.py
from __future__ import annotations

import numpy as np
import cv2

# ---- 設定（config.py が無い/未定義でも動くようフォールバック） ----
try:
    from ..config import (
        OSNET_INPUT_SIZE_HW,        # (H, W) 例: (256, 128)
        OSNET_MIN_CROP_SIDE,     # 例: 48
        OSNET_PAD_VALUE,         # 例: (114,114,114)
        REID_TTA_HFLIP,          # True/False
    )
except Exception:
    OSNET_INPUT_SIZE_HW    = (256, 128)
    OSNET_MIN_CROP_SIDE = 48
    OSNET_PAD_VALUE     = (114, 114, 114)
    REID_TTA_HFLIP      = False


# ========== 前処理ユーティリティ ==========
def ensure_min_short_side(bgr: np.ndarray, min_short: int = OSNET_MIN_CROP_SIDE) -> np.ndarray:
    """短辺が min_short 未満なら等倍率で拡大（CUBIC）。"""
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    s = min(h, w)
    if s >= min_short or s <= 0:
        return bgr
    scale = float(min_short) / float(s)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)


def letterbox_resize(bgr: np.ndarray, target_hw: tuple[int, int] = OSNET_INPUT_SIZE_HW,
                     pad_value: tuple[int, int, int] = OSNET_PAD_VALUE) -> np.ndarray:
    """
    アスペクト比保持で target(H,W) へリサイズ + 中央パディング（レターボックス）。
    """
    Ht, Wt = target_hw
    if bgr is None or bgr.size == 0:
        return np.full((Ht, Wt, 3), pad_value, dtype=np.uint8)

    ch, cw = bgr.shape[:2]
    r = min(Ht / float(ch), Wt / float(cw))
    new_h, new_w = int(round(ch * r)), int(round(cw * r))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top   = (Ht - new_h) // 2
    pad_bot   = Ht - new_h - pad_top
    pad_left  = (Wt - new_w) // 2
    pad_right = Wt - new_w - pad_left

    out = cv2.copyMakeBorder(
        resized, pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_value
    )
    return out


def imagenet_norm_to_chw(img_bgr: np.ndarray) -> np.ndarray:
    """BGR→RGB, [0,1] 正規化, ImageNet標準化, CHW, float32"""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("imagenet_norm_to_chw: empty input")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    chw = np.transpose(img, (2, 0, 1))  # (3,H,W)
    return chw.astype(np.float32)


def preprocess_osnet(person_bgr: np.ndarray,
                     input_hw: tuple[int, int] = OSNET_INPUT_SIZE_HW,
                     min_short: int = OSNET_MIN_CROP_SIDE) -> np.ndarray:
    """
    OSNet前処理：短辺底上げ→レターボックス→ImageNet正規化→CHW→NCHW(batch=1)
    戻り値: (1,3,H,W) float32
    """
    x = ensure_min_short_side(person_bgr, min_short)
    x = letterbox_resize(x, input_hw, OSNET_PAD_VALUE)
    x = imagenet_norm_to_chw(x)[None, ...]
    return x


# ========== 推論＆正規化ユーティリティ ==========
def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = vec.astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(v)) + eps
    return (v / n).astype(np.float32)


def _onnx_forward(session, inp_name: str, nchw: np.ndarray) -> np.ndarray:
    out = session.run(None, {str(inp_name): nchw})[0]
    v = np.asarray(out).reshape(-1).astype(np.float32)
    return v


def tta_embed(session, inp_name: str, person_bgr: np.ndarray, use_hflip: bool = REID_TTA_HFLIP) -> np.ndarray:
    """
    1枚のクロップに対して TTA（水平フリップ平均）を適用して L2正規化ベクトルを返す。
    """
    X = preprocess_osnet(person_bgr)
    v = _onnx_forward(session, inp_name, X)

    if not use_hflip:
        return l2_normalize(v)

    flipped = cv2.flip(person_bgr, 1)
    Xf = preprocess_osnet(flipped)
    vf = _onnx_forward(session, inp_name, Xf)

    return l2_normalize(0.5 * (v + vf))


def embed_batch(session, inp_name: str, crops_bgr: list[np.ndarray],
                use_hflip: bool = REID_TTA_HFLIP, batch_size: int = 16) -> np.ndarray:
    """
    複数のクロップをまとめて推論。TTA時は normal/flip を平均 → L2正規化。
    戻り値: (N, D) float32
    """
    if not crops_bgr:
        return np.zeros((0, 512), np.float32)

    # まとめて前処理（NCHW）をバッチ送信
    feats: list[np.ndarray] = []
    N = len(crops_bgr)
    for i in range(0, N, max(1, int(batch_size))):
        chunk = crops_bgr[i:i+batch_size]
        X = np.concatenate([preprocess_osnet(c) for c in chunk], axis=0)  # (B,3,H,W)
        v = session.run(None, {str(inp_name): X})[0]                      # (B,D)
        v = v.astype(np.float32)

        if use_hflip:
            Xf = np.concatenate([preprocess_osnet(cv2.flip(c, 1)) for c in chunk], axis=0)
            vf = session.run(None, {str(inp_name): Xf})[0].astype(np.float32)
            v = 0.5 * (v + vf)  # 平均

        # L2正規化を行列方向に
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        v = (v / n).astype(np.float32)
        feats.append(v)

    return np.concatenate(feats, axis=0)
