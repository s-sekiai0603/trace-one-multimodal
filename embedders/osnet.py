# osnet.py — ReID 推論本体（Identifierから委譲される）＋ ギャラリー実装
from __future__ import annotations
import os
import json
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Callable
import numpy as np
import logging


from ..utils.osnet_utils import (
    preprocess_osnet, tta_embed, embed_batch as utils_embed_batch
)
from ..utils.preprocess_video import VideoPreprocessor
from .. import config
from ..config import MIN_CROP_SIDE, REID_TTA_HFLIP, REID_BATCH_SIZE

# 依存（任意）
try:
    import onnxruntime as ort
except ImportError:
    ort = None
    
log = logging.getLogger("app.osnet")

# try:
#     import faiss  # pip install faiss-cpu
# except ImportError:
#     faiss = None

# if TYPE_CHECKING:
#     import faiss as _faiss
#     FaissIndex = _faiss.Index
# else:
#     FaissIndex = object


# ========= ユーティリティ =========
def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ========= 本体：Identifier から状態を受け取って処理する仲介クラス =========
class Osnet:
    """
    Identifier 側の状態（session, inp_name, input_size, output_dim）にアクセスする
    コールバックを受け取り、_preprocess / embed / embed_batch を実装する。
    """
    def __init__(
        self,
        get_state: Callable[[], Dict[str, object]],
        set_output_dim: Callable[[int], None],
    ):
        """
        get_state() は dict を返す必要があります:
          {
            "session": onnxruntime.InferenceSession | None,
            "inp_name": str | None,
            "input_size": Tuple[int, int],
            "output_dim": int | None,
          }
        """
        self._get_state = get_state
        self._set_output_dim = set_output_dim
        
        # ★ 外見識別専用の前処理（VideoPreprocessor）
        self._preproc: Optional[VideoPreprocessor]
        try:
            vp = VideoPreprocessor(config.__dict__)
            if getattr(vp, "enabled", False):
                self._preproc = vp
                log.info("[OSNET][PREPROC] enabled for appearance embeddings.")
            else:
                self._preproc = None
        except Exception as e:
            log.warning("[OSNET][PREPROC] disabled: %s", e)
            self._preproc = None

    # # Identifier と同名のメソッドとして実装（呼び出し元は変えない）
    # def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
    #     """
    #     ReID前処理：
    #     - アスペクト比保持で target(H,W) にレターボックス
    #     - 短辺が MIN_CROP_SIDE 未満なら底上げ
    #     - ImageNet正規化 → CHW(N=1)
    #     """
    #     state = self._get_state()
    #     Ht, Wt = state["input_size"]  # (H, W) 例: (256,128)
    #     if bgr is None or bgr.size == 0:
    #         import numpy as _np
    #         blank = _np.zeros((Ht, Wt, 3), _np.uint8)
    #         img = blank
    #     else:
    #         import cv2
    #         ch, cw = bgr.shape[:2]

    #         # 1) 短辺の最低値を確保（情報不足の回避）
    #         try:
    #             min_short = int(MIN_CROP_SIDE)
    #         except Exception:
    #             min_short = 48
    #         if min(ch, cw) < min_short:
    #             scale = float(min_short) / float(max(1, min(ch, cw)))
    #             nh, nw = int(round(ch * scale)), int(round(cw * scale))
    #             bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    #             ch, cw = bgr.shape[:2]

    #         # 2) 等比スケーリング → レターボックス（パディングは114系）
    #         r = min(Ht / float(ch), Wt / float(cw))
    #         new_h, new_w = int(round(ch * r)), int(round(cw * r))
    #         resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    #         pad_top  = (Ht - new_h) // 2
    #         pad_bot  = Ht - new_h - pad_top
    #         pad_left = (Wt - new_w) // 2
    #         pad_right= Wt - new_w - pad_left
    #         img = cv2.copyMakeBorder(resized, pad_top, pad_bot, pad_left, pad_right,
    #                                 borderType=cv2.BORDER_CONSTANT, value=(114,114,114))

    #     # 3) RGB化 → [0,1] → ImageNet正規化 → CHW(N=1)
    #     import numpy as np
    #     import cv2
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    #     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    #     std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #     img = (img - mean) / std
    #     chw = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,H,W)
    #     return chw

    def embed(self, person_bgr: np.ndarray) -> np.ndarray:
        state = self._get_state()
        session = state["session"]; inp_name = state["inp_name"]
        
        if self._preproc is not None and person_bgr is not None:
            try:
                person_bgr = self._preproc.apply(person_bgr)
            except Exception as e:
                log.warning("[OSNET][PREPROC] apply failed (single): %s", e)
                
        # 新: ユーティリティでまとめて TTA + L2 正規化
        return tta_embed(session, inp_name, person_bgr, use_hflip=REID_TTA_HFLIP)


    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        state = self._get_state()
        session = state["session"]; inp_name = state["inp_name"]
        
        proc_crops = crops_bgr
        if self._preproc is not None and crops_bgr:
            proc_crops = []
            for c in crops_bgr:
                if c is None:
                    proc_crops.append(c)
                    continue
                try:
                    proc_crops.append(self._preproc.apply(c))
                except Exception as e:
                    log.warning("[OSNET][PREPROC] apply failed (batch): %s", e)
                    proc_crops.append(c)
                    
        return utils_embed_batch(
            session, inp_name,
            crops_bgr,
            use_hflip=REID_TTA_HFLIP,
            batch_size=REID_BATCH_SIZE
        )

# ========= ギャラリー: JSON 版（旧 JsonIdentityGallery） =========
class OsnetJsonGallery:
    """
    FAISS版 IdentityGallery と互換APIを提供する JSON バックエンド実装。
    - base_dir/labels.json に永続化
    - ベクトルは float32, shape=(D,) 前提（必要なら内部で矯正）
    - add() で True を返したときのみギャラリが更新されたとみなす
    """
    def __init__(self, base_dir: str, dim: Optional[int] = None) -> None:
        self.base_dir = str(base_dir)
        self.meta_path = os.path.join(self.base_dir, "labels.json")
        os.makedirs(self.base_dir, exist_ok=True)

        self._dim: Optional[int] = int(dim) if dim else None
        # メモリ上の構造: {label: [np.ndarray(D,), ...]}
        self._label_to_vecs: Dict[str, List[np.ndarray]] = {}

        # 既存があればロード
        if os.path.isfile(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._dim = int(data.get("dim") or 0) or self._dim
                lab2 = data.get("label_to_vecs") or {}
                for lab, vecs in lab2.items():
                    self._label_to_vecs[str(lab)] = [
                        np.asarray(v, dtype=np.float32).reshape(-1)
                        for v in (vecs or [])
                    ]
                # 形の安全確認
                if self._dim is None:
                    # 何か1つでもあれば次元を確定
                    for _lab, arrs in self._label_to_vecs.items():
                        if arrs:
                            self._dim = int(arrs[0].shape[0])
                            break
            except Exception:
                # 壊れていたら空で始める
                self._dim = int(dim) if dim else None
                self._label_to_vecs = {}

    # 互換API: 件数
    @property
    def ntotal(self) -> int:
        return sum(len(vs) for vs in self._label_to_vecs.values())

    # 互換API: ラベル一覧
    def all_labels(self) -> List[str]:
        return list(self._label_to_vecs.keys())

    # 互換API: 追加
    def add(self, label: str, vec: np.ndarray) -> bool:
        """
        P-ID の特徴量を 1 件追加する。
        - メモリ(_label_to_vecs)へ格納
        - 直ちに labels.jsonl へ 1 行追記（{"label": "...", "vec": [...]}）
        - 次元不一致は拒否
        """
        if not label:
            return False

        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if self._dim is None:
            self._dim = int(v.shape[0])
        if int(v.shape[0]) != int(self._dim):
            return False  # 次元不一致は登録しない

        # メモリへ反映
        vs = self._label_to_vecs.setdefault(str(label), [])
        vs.append(v.copy())

        # 追記（1行=1特徴量、同一P-IDは複数行になる）
        try:
            jsonl_path = os.path.join(self.base_dir, "labels.jsonl")
            line_obj = {"label": str(label), "vec": v.tolist()}
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line_obj, ensure_ascii=False))
                f.write("\n")
        except Exception:
            # 追記失敗でも add 自体は成功扱い（メモリは更新済み）
            pass

        return True


    # 互換API: センロイド（平均ベクトル）
    def centroid(self, label: str) -> Optional[np.ndarray]:
        arrs = self._label_to_vecs.get(str(label))
        if not arrs:
            return None
        mat = np.vstack(arrs).astype(np.float32, copy=False)  # (N,D)
        cen = mat.mean(axis=0)
        nrm = float(np.linalg.norm(cen)) + 1e-12
        return (cen / nrm).astype(np.float32, copy=False)

    # 互換API: 近傍検索（簡易：全件コサインで topk）
    def search(self, q: np.ndarray, topk: int = 5) -> List[Tuple[str, float]]:
        if self.ntotal == 0:
            return []
        qv = np.asarray(q, dtype=np.float32).reshape(-1)
        nrm = float(np.linalg.norm(qv)) + 1e-12
        qv = (qv / nrm).astype(np.float32, copy=False)
        # ラベルのセンロイドで検索（高速・安定）
        outs: List[Tuple[str, float]] = []
        for lab in self._label_to_vecs.keys():
            cen = self.centroid(lab)
            if cen is None:
                continue
            sim = float(np.dot(qv, cen))
            outs.append((lab, sim))
        # 類似度降順 topk
        outs.sort(key=lambda t: t[1], reverse=True)
        return outs[: int(topk or 5)]

    # 互換API: 保存
    def save(self) -> None:
        """
        永続化（安全に再生成）。
        - labels.jsonl : 1行=1特徴量（同一P-IDは複数行）
                        例: {"label":"P0001","vec":[...]}
        - labels.json  : 互換用スナップショット（全サンプルを集約）
                        {"dim":D,"label_to_vecs":{"P0001":[[...], ...], ...}}
        """
        # ---- 1) JSONL をメモリから完全再生成（壊れや重複に強くする）----
        jsonl_path = os.path.join(self.base_dir, "labels.jsonl")
        tmp_jsonl = jsonl_path + ".tmp"
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for lab, vecs in self._label_to_vecs.items():
                for v in vecs:
                    v1 = np.asarray(v, dtype=np.float32).reshape(-1)
                    line_obj = {"label": str(lab), "vec": v1.tolist()}
                    f.write(json.dumps(line_obj, ensure_ascii=False))
                    f.write("\n")
        os.replace(tmp_jsonl, jsonl_path)

        # ---- 2) 集約JSON（可視化・再起動ロード用）----
        snapshot = {
            "dim": int(self._dim or 0),
            "label_to_vecs": {
                lab: [np.asarray(v, dtype=np.float32).reshape(-1).tolist() for v in vecs]
                for lab, vecs in self._label_to_vecs.items()
            },
        }
        json_path = os.path.join(self.base_dir, "labels.json")
        tmp_json = json_path + ".tmp"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp_json, json_path)
