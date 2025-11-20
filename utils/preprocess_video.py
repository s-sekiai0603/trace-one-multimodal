# -*- coding: utf-8 -*-
"""
動画前処理（BGR画像に対して逐次適用）
- 明るさ/コントラスト
- ガンマ補正
- ヒストグラム平坦化（CLAHE, Yチャネル）
- ノイズ除去（fastNlMeansDenoisingColored）
- シャープネス（アンシャープマスク）
- 超解像（cv2.dnn_superres が使えれば / 無ければバイリニア/バイキュービック）
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import cv2

try:
    from cv2 import dnn_superres  # type: ignore
    _HAS_SR = True
except Exception:
    _HAS_SR = False

class VideoPreprocessor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get("PREPROCESS", {})
        self.enabled = bool(self.cfg.get("ENABLE", False))
        self.order = list(self.cfg.get("ORDER",
            ["BRICON","GAMMA","HIST_EQ","DENOISE","SHARPEN","SUPERRES"]))

        # super-res 初期化
        self._sr = None
        sr_cfg = self.cfg.get("SUPERRES", {})
        if self.enabled and sr_cfg.get("ENABLE", False):
            if _HAS_SR:
                try:
                    self._sr = dnn_superres.DnnSuperResImpl_create()
                    model_path = sr_cfg.get("MODEL_PATH", "")
                    engine = str(sr_cfg.get("ENGINE", "edsr")).lower()  # edsr|lapsrn|espcn|fsrcnn など
                    scale = int(sr_cfg.get("SCALE", 2))
                    if model_path:
                        self._sr.readModel(model_path)
                        self._sr.setModel(engine, scale)
                except Exception:
                    self._sr = None  # フォールバックに任せる

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        if not self.enabled or bgr is None:
            return bgr

        out = bgr

        for step in self.order:
            if step == "BRICON" and self.cfg.get("BRICON", {}).get("ENABLE", True):
                p = self.cfg["BRICON"]
                out = self._brightness_contrast(out, alpha=float(p.get("ALPHA", 1.15)), beta=float(p.get("BETA", 8)))

            elif step == "GAMMA" and self.cfg.get("GAMMA", {}).get("ENABLE", True):
                g = float(self.cfg["GAMMA"].get("GAMMA", 0.95))
                out = self._gamma(out, gamma=g)

            elif step == "HIST_EQ" and self.cfg.get("HIST_EQ", {}).get("ENABLE", True):
                p = self.cfg["HIST_EQ"]
                out = self._clahe_y(out, clip=float(p.get("CLAHE_CLIP", 2.0)), tile=int(p.get("CLAHE_TILE", 8)))

            elif step == "DENOISE" and self.cfg.get("DENOISE", {}).get("ENABLE", True):
                p = self.cfg["DENOISE"]
                out = cv2.fastNlMeansDenoisingColored(
                    out, None,
                    h=float(p.get("H", 3.0)),
                    hColor=float(p.get("HCOLOR", 3.0)),
                    templateWindowSize=int(p.get("TEMPLATE_WINDOW", 7)),
                    searchWindowSize=int(p.get("SEARCH_WINDOW", 21))
                )

            elif step == "SHARPEN" and self.cfg.get("SHARPEN", {}).get("ENABLE", True):
                p = self.cfg["SHARPEN"]
                out = self._unsharp(
                    out,
                    sigma=float(p.get("SIGMA", 1.2)),
                    amount=float(p.get("AMOUNT", 1.0)),
                    threshold=float(p.get("THRESHOLD", 3.0)),
                )

            elif step == "SUPERRES" and self.cfg.get("SUPERRES", {}).get("ENABLE", False):
                out = self._superres(out)

        return out

    @staticmethod
    def _brightness_contrast(img: np.ndarray, alpha: float = 1.15, beta: float = 8.0) -> np.ndarray:
        # y = alpha * x + beta
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def _gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        gamma = max(1e-6, float(gamma))
        lut = (np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0).astype(np.uint8)
        return cv2.LUT(img, lut)

    @staticmethod
    def _clahe_y(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
        y2 = clahe.apply(y)
        return cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _unsharp(img: np.ndarray, sigma: float = 1.2, amount: float = 1.0, threshold: float = 3.0) -> np.ndarray:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # アンシャープ: img * (1+amount) - blur * amount
        sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
        if threshold > 0:
            low_contrast_mask = (cv2.absdiff(img, blur) < threshold).astype(np.uint8)
            sharp = img * low_contrast_mask + sharp * (1 - low_contrast_mask)
            sharp = sharp.astype(np.uint8)
        return sharp

    def _superres(self, img: np.ndarray) -> np.ndarray:
        # dnn_superres が使えればそれを、なければCUBICアップスケール
        sr_cfg = self.cfg.get("SUPERRES", {})
        scale = int(sr_cfg.get("SCALE", 2))
        if self._sr is not None:
            try:
                return self._sr.upsample(img)
            except Exception:
                pass
        # フォールバック
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
