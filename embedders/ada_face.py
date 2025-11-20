# -*- coding: utf-8 -*-
"""
AdaFace (PyTorch, .ckpt) で顔特徴量を抽出するラッパー。
- 入力: BGRフレーム (H,W,3), BBOX (x1,y1,x2,y2[,score])
- 前処理: 余白付与で切り出し→112x112→RGB, [-1,1] 正規化→CHW→Tensor
- 出力: L2正規化済み埋め込み (D,)  ※典型は D=512
"""
from __future__ import annotations
from typing import Optional, List, Tuple
import logging
import numpy as np
import cv2
import torch

log = logging.getLogger("app.adaface")

# デバイス既定（config があれば優先）
try:
    from ..config import  (
        TORCH_DEVICE, ADAFACE_CKPT_PATH, ADAFACE_ARCH, ADAFACE_MARGIN,
        ADAFACE_OUT_SIZE, ADAFACE_LOG_EVERY
    )
except Exception:
    TORCH_DEVICE = "cpu"
    ADAFACE_CKPT_PATH = "models/adaface_ir101_webface12m.ckpt"
    ADAFACE_ARCH = "ir_101"
    ADAFACE_MARGIN = 0.20
    ADAFACE_OUT_SIZE = 112
    ADAFACE_LOG_EVERY = 10

try:
    from . import net  # AdaFace 公式実装の net.py をプロジェクト直下に配置しておくこと
except Exception as e:
    raise ImportError("`net.py` が見つかりません。AdaFace の公式実装をプロジェクト直下に配置してください。") from e


class AdaFaceEmbedder:
    def __init__(
        self,
        ckpt_path: str = ADAFACE_CKPT_PATH,
        architecture: str = ADAFACE_ARCH,
        device: str = TORCH_DEVICE,
        margin: float = ADAFACE_MARGIN,
        out_size: int = ADAFACE_OUT_SIZE,
        log_each: int = ADAFACE_LOG_EVERY,
    ) -> None:
        self.log = logging.getLogger("app.adaface")
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        self.architecture = architecture
        self.margin = float(margin)
        self.out_size = int(out_size)
        self._count = 0
        self._log_each = max(1, int(log_each))

        # ---- モデル構築 & 重みロード ----
        self.model = net.build_model(self.architecture).to(self.device).eval()

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)

        # 'model.' プレフィクスを取り除いてロード
        model_sd = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.") or k in state_dict}
        missing, unexpected = self.model.load_state_dict(model_sd, strict=False)
        self.log.info(
            f"[AdaFace] loaded ckpt='{self.ckpt_path}' arch={self.architecture} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

        # 特徴次元の推定（失敗時は 512 とみなす）
        self.feat_dim = 512
        try:
            for k, v in self.model.state_dict().items():
                if k.endswith("output_layer.4.weight"):  # Linear(out_dim, in_dim)
                    self.feat_dim = int(v.shape[0])
                    break
        except Exception:
            pass
        self.log.info(f"[AdaFace] feature_dim={self.feat_dim}, device={self.device.type}")

    # ========= ユーティリティ =========
    @staticmethod
    def _clip_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
        return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)

    def _crop_with_margin(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(float, box_xyxy[:4])
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # 余白を付与
        mx = self.margin * w
        my = self.margin * h
        x1 -= mx; y1 -= my; x2 += mx; y2 += my
        x1, y1, x2, y2 = self._clip_box(int(x1), int(y1), int(x2), int(y2), W, H)

        crop = frame_bgr[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
        return crop

    @staticmethod
    def _to_tensor_adaface(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        BGR → RGB → [-1,1] → CHW → Tensor[1,3,112,112]
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5  # [-1,1]
        chw = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=torch.float32)

    @staticmethod
    def _l2norm_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
        return x / n

    # ========= 推論 =========
    def embed_one(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> Optional[np.ndarray]:
        """
        単一BBOXから埋め込み (D,) を返す。失敗時 None。
        """
        crop = self._crop_with_margin(frame_bgr, box_xyxy)
        if crop is None:
            self.log.debug("[AdaFace] crop failed (empty).")
            return None

        inp = self._to_tensor_adaface(crop, self.device)

        with torch.no_grad():
            feat, _norm = self.model(inp)  # AdaFace は (feature, norm) を返す実装
        feat_np = feat.detach().cpu().numpy().astype(np.float32)[0]
        feat_np = self._l2norm_np(feat_np)

        # ログ（多すぎ防止）
        self._count += 1
        if self._count % self._log_each == 0:
            self.log.info(f"[AdaFace] feat#{self._count}: shape={feat_np.shape}, "
                          f"||f||≈{float(np.linalg.norm(feat_np)):.3f}")

        return feat_np

    def embed_many(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        複数BBOX → (N, D)。失敗行は 0 ベクトル。
        """
        if boxes_xyxy is None:
            return np.zeros((0, self.feat_dim), np.float32)

        boxes = np.asarray(boxes_xyxy)
        if boxes.ndim == 1:
            boxes = boxes[None, :]  # (4,) → (1,4)
        if boxes.shape[0] == 0:
            return np.zeros((0, self.feat_dim), np.float32)

        feats: List[np.ndarray] = []
        for b in boxes:  # b は (4,) 想定
            f = self.embed_one(frame_bgr, b)
            if f is None:
                feats.append(np.zeros((self.feat_dim,), np.float32))
            else:
                feats.append(f.astype(np.float32, copy=False))

        return np.vstack(feats).astype(np.float32)
