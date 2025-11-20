# -*- coding: utf-8 -*-
# evidence_capture.py
from __future__ import annotations
import os, time, uuid
import cv2
import numpy as np
from typing import Optional

class EvidenceCapture:
    """
    ・識別ON中だけフレームをカウントし、start_afterフレーム目で1枚、
      以後 interval フレームごとにBGR画像をPNG保存。
    ・保存先: ./evidence/capture/<MMDDHHMMSS>/evidence_<N>.png
    """
    def __init__(
        self,
        start_after: int = 32,
        interval: int = 2,
        save_root: str = os.path.join("evidence", "capture"),
        file_prefix: str = "evidence_",
        ext: str = ".png",
    ):
        self.start_after = int(start_after)
        self.interval = int(interval)
        self.save_root = save_root
        self.file_prefix = file_prefix
        self.ext = ext

        self._session_dir: Optional[str] = None
        self._save_idx: int = 0

    # ---- セッション制御 ----
    def start_new_session(self) -> str:
        ts = time.strftime("%m%d%H%M%S", time.localtime())
        root = os.path.abspath(os.getcwd())
        out_dir = os.path.join(root, self.save_root, ts)
        os.makedirs(out_dir, exist_ok=True)
        self._session_dir = out_dir
        self._save_idx = 0
        return out_dir

    # ---- 保存コア ----
    def _unicode_safe_save_png(self, bgr: np.ndarray, path: str) -> bool:
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return False
        tmp = f"{path}.tmp.{uuid.uuid4().hex}.png"
        try:
            buf.tofile(tmp)     # UnicodeパスOK
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False

    # ---- 外部から呼ぶAPI（最小・同期版）----
    def save_if_needed(self, frame_idx: int, active: bool, bgr_frame: np.ndarray) -> Optional[str]:
        """
        識別ON(active=True)かつ frame_idx が:
          - start_after ちょうど
          - 以後は interval ごと
        のときに保存。保存したらファイルパスを返す。未保存は None。
        """
        if not active or bgr_frame is None or bgr_frame.size == 0:
            return None
        if frame_idx < self.start_after:
            return None
        if frame_idx != self.start_after and ((frame_idx - self.start_after) % self.interval != 0):
            return None

        if self._session_dir is None:
            # 想定外だが安全側：セッション自動生成
            self.start_new_session()

        self._save_idx += 1
        fname = f"{self.file_prefix}{self._save_idx}{self.ext}"
        path = os.path.join(self._session_dir, fname)
        ok = self._unicode_safe_save_png(bgr_frame, path)
        return path if ok else None
