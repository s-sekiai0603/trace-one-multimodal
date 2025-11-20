# gait_utils.py
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, List, Tuple

from ..config import (
    NORM_CENTER_HIPS,
    NORM_SCALE_SHOULDERS,
    NORM_HEIGHT_TOP_IDX,
    NORM_HEIGHT_BOTTOM_IDXS,
    NORM_MIN_SCALE,
    NORM_EWMA_ENABLED,
    NORM_EWMA_ALPHA,
    NORM_EWMA_APPLY_TO_CENTER,
    NORM_EWMA_APPLY_TO_SCALE,
    KPT_VALID_MIN,
    COCO_LR_PAIRS,
    FLIP_TTA_L2_NORMALIZE,
    NOISE_TTA_DEFAULT_SIGMA,
    NOISE_TTA_DEFAULT_SAMPLES,
    NOISE_TTA_INCLUDE_BASE,
    NOISE_TTA_L2_NORMALIZE,
    TTA_TEMPORAL_L2_NORMALIZE,
    POSE_GATE_ENABLED,
    POSE_GATE_MIN_CONF,
    POSE_GATE_LOWER_MIN,
    POSE_GATE_SIDE_MIN,
    HYBRID_TTA_ENABLED,
    HYBRID_TTA_ALPHA,
    TTA_MODES,
    TTA_SCORE_TOPK
)


class GaitUtils:
    """
    Gait用ユーティリティクラス。
    ・normalize_pose(): 骨格正規化（方法B）
    ・flip_tta_embed(): 左右反転TTA＋平均化
    ・noise_tta_embed(): 軽微ノイズTTA
    """

    # ----------------------------------------
    # 骨格正規化（方法B）関連メソッド
    # ----------------------------------------
    def normalize_pose(self, seq: np.ndarray) -> np.ndarray:
        """
        骨格正規化ワークフロー：
        ①中心化 → ②スケーリング → ③（任意）時系列の安定化（EMA）
        入力：seq (T, K, 2)
        出力：正規化後 seq_norm (T, K, 2)
        """
        a = np.asarray(seq, dtype=np.float32)
        if a.ndim != 3 or a.shape[-1] != 2:
            raise ValueError("normalize_pose: expected shape (T, K, 2).")

        T, K, _ = a.shape
        out = a.copy()

        # フレーム別 center/scale 計算
        centers = np.zeros((T, 2), dtype=np.float32)
        scales = np.ones((T,), dtype=np.float32)

        for t in range(T):
            kpts = out[t]
            centers[t] = self._frame_center(kpts)
            scales[t] = max(self._frame_scale(kpts), NORM_MIN_SCALE)

        # 時系列の安定化（EMA）
        if NORM_EWMA_ENABLED and T >= 2:
            alpha = float(NORM_EWMA_ALPHA)
            if NORM_EWMA_APPLY_TO_CENTER:
                prev_c = centers[0].copy()
                for t in range(1, T):
                    centers[t] = alpha * centers[t] + (1.0 - alpha) * prev_c
                    prev_c = centers[t]
            if NORM_EWMA_APPLY_TO_SCALE:
                prev_s = scales[0]
                for t in range(1, T):
                    scales[t] = alpha * scales[t] + (1.0 - alpha) * prev_s
                    prev_s = scales[t]

        # 適用
        for t in range(T):
            out[t, :, 0] = (out[t, :, 0] - centers[t, 0]) / scales[t]
            out[t, :, 1] = (out[t, :, 1] - centers[t, 1]) / scales[t]

        return out

    def _frame_center(self, kpts: np.ndarray) -> np.ndarray:
        """フレーム単位：ヒップ中心 or有効点平均"""
        li, ri = NORM_CENTER_HIPS
        cand = []
        if self._is_valid_point(kpts, li):
            cand.append(kpts[li])
        if self._is_valid_point(kpts, ri):
            cand.append(kpts[ri])
        if cand:
            return np.mean(np.stack(cand, axis=0), axis=0).astype(np.float32)

        mask = self._valid_mask(kpts)
        if mask.any():
            return np.mean(kpts[mask], axis=0).astype(np.float32)

        return np.zeros(2, dtype=np.float32)

    def _frame_scale(self, kpts: np.ndarray) -> float:
        """フレーム単位：肩幅基準 or身長代替 orフォールバック1.0"""
        ls, rs = NORM_SCALE_SHOULDERS
        if self._is_valid_point(kpts, ls) and self._is_valid_point(kpts, rs):
            d = float(np.linalg.norm(kpts[ls] - kpts[rs]))
            if d > 0.0:
                return d

        top = NORM_HEIGHT_TOP_IDX
        bottoms = []
        for idx in NORM_HEIGHT_BOTTOM_IDXS:
            if self._is_valid_point(kpts, idx):
                bottoms.append(kpts[idx])
        if self._is_valid_point(kpts, top) and bottoms:
            foot = np.mean(np.stack(bottoms, axis=0), axis=0)
            d = float(np.linalg.norm(kpts[top] - foot))
            if d > 0.0:
                return d

        return 1.0

    def _is_valid_point(self, kpts: np.ndarray, idx: int) -> bool:
        x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
        return (abs(x) + abs(y) > KPT_VALID_MIN) and np.isfinite(x) and np.isfinite(y)

    def _valid_mask(self, kpts: np.ndarray) -> np.ndarray:
        xy_sum = np.abs(kpts).sum(axis=1)
        return (xy_sum > KPT_VALID_MIN) & np.isfinite(kpts).all(axis=1)

    # ----------------------------------------
    # 左右反転TTA＋平均化
    # ----------------------------------------
    def flip_tta_embed(self, seq: np.ndarray, embed_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        ①反転骨格生成 → ②2系統埋め込み → ③平均化 →（オプション）L2正規化
        入力：seq (T, K, 2)、embed_fn: seq→(D,)
        出力：feat_mean (D,)
        """
        a = np.asarray(seq, dtype=np.float32)
        if a.ndim != 3 or a.shape[-1] != 2:
            raise ValueError("flip_tta_embed: expected shape (T, K, 2).")

        # 反転生成
        flipped = a.copy()
        flipped[..., 0] = -flipped[..., 0]
        for li, ri in COCO_LR_PAIRS:
            flipped[:, [li, ri], :] = flipped[:, [ri, li], :]

        # 埋め込み取得
        f_orig = self._to_numpy_1d(embed_fn(a))
        f_flip = self._to_numpy_1d(embed_fn(flipped))

        feat = (f_orig + f_flip) * 0.5
        if FLIP_TTA_L2_NORMALIZE:
            feat = self._l2_normalize(feat)
        return feat

    # ----------------------------------------
    # 軽微ノイズTTA
    # ----------------------------------------
    def noise_tta_embed(
        self,
        seq: np.ndarray,
        embed_fn: Callable[[np.ndarray], np.ndarray],
        sigma: Optional[float] = None,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        ①原系列埋め込み → ②ノイズ付き系列n回 → ③平均化 →（オプション）L2正規化
        入力：seq (T, K, 2)、embed_fn、sigma、n_samples
        出力：feat_mean (D,)
        """
        a = np.asarray(seq, dtype=np.float32)
        if a.ndim != 3 or a.shape[-1] != 2:
            raise ValueError("noise_tta_embed: expected shape (T, K, 2).")

        sig = float(NOISE_TTA_DEFAULT_SIGMA if sigma is None else sigma)
        m = int(NOISE_TTA_DEFAULT_SAMPLES if n_samples is None else n_samples)

        feats: List[np.ndarray] = []

        if NOISE_TTA_INCLUDE_BASE:
            feats.append(self._to_numpy_1d(embed_fn(a)))

        for _ in range(max(0, m)):
            noise = np.random.normal(0.0, sig, size=a.shape).astype(np.float32)
            seq_noisy = a + noise
            feats.append(self._to_numpy_1d(embed_fn(seq_noisy)))

        if not feats:
            raise RuntimeError("noise_tta_embed: no features generated.")

        feat = np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32, copy=False)
        if NOISE_TTA_L2_NORMALIZE:
            feat = self._l2_normalize(feat)
        return feat
    
    # ----------------------------------------
    # 時間反転TTA（順方向と逆順の2系列を平均）
    # ----------------------------------------
    def temporal_reverse_tta_embed(
        self,
        seq: np.ndarray,
        embed_fn: Callable[[np.ndarray], np.ndarray],
        l2_normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        ① 正方向系列 → 埋め込み
        ② 逆順系列   → 埋め込み
        ③ 平均（オプションでL2正規化）
        入力:  seq (T, K, 2)
        出力:  (D,)
        """
        a = np.asarray(seq, dtype=np.float32)
        if a.ndim != 3 or a.shape[-1] != 2:
            raise ValueError("temporal_reverse_tta_embed: expected shape (T, K, 2).")

        f_fwd = self._to_numpy_1d(embed_fn(a))
        f_rev = self._to_numpy_1d(embed_fn(a[::-1]))

        feat = (f_fwd + f_rev) * 0.5
        use_l2 = TTA_TEMPORAL_L2_NORMALIZE if l2_normalize is None else bool(l2_normalize)
        if use_l2:
            feat = self._l2_normalize(feat)
        return feat

    # ----------------------------------------
    # ヘルパメソッド
    # ----------------------------------------
    def _to_numpy_1d(self, x) -> np.ndarray:
        """embed_fn出力を (D,) np.float32 に変換"""
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except ImportError:
            pass
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        return arr

    def _l2_normalize(self, v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        if norm < eps:
            return v.astype(np.float32, copy=False)
        return (v / (norm + eps)).astype(np.float32, copy=False)

    def pose_passes_gate(self, kpts_k3: np.ndarray) -> bool:
        """
        フレーム内の1人分キーポイント (K,3) に対して品質ゲートをかける。
        True: 学習/識別の時系列へ採用, False: スキップ
        ルール:
         - conf>=POSE_GATE_MIN_CONF かつ xy が有限
         - 下半身(11..16)の有効点合計 >= POSE_GATE_LOWER_MIN
         - 膝or足首が左右それぞれに1点以上
         - 左右の半身(肩~手首/腰~足首)の有効点が双方 >= POSE_GATE_SIDE_MIN
        """
        if not bool(POSE_GATE_ENABLED):
            return True
        a = np.asarray(kpts_k3, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] < 2:
            return False

        K = a.shape[0]
        # conf が無い入力への保険（全点 conf=1 とみなす）
        if a.shape[1] >= 3:
            conf = a[:, 2]
        else:
            conf = np.ones((K,), dtype=np.float32)

        x = a[:, 0]; y = a[:, 1]
        valid = (conf >= float(POSE_GATE_MIN_CONF)) & np.isfinite(x) & np.isfinite(y)

        # インデックス集合（COCO-17）
        LB = np.array([11,12,13,14,15,16])                  # lower body
        LEFT  = np.array([5,7,9,11,13,15])                  # 左半身
        RIGHT = np.array([6,8,10,12,14,16])                 # 右半身

        # 下半身の総数
        lb_count = int(np.sum(valid[LB]))
        if lb_count < int(POSE_GATE_LOWER_MIN):
            return False

        # 膝/足首の左右存在
        left_lower_ok  = bool(valid[13] or valid[15])  # left knee or ankle
        right_lower_ok = bool(valid[14] or valid[16])  # right knee or ankle
        if not (left_lower_ok and right_lower_ok):
            return False

        # 左右の半身バランス
        left_count  = int(np.sum(valid[LEFT]))
        right_count = int(np.sum(valid[RIGHT]))
        if left_count < int(POSE_GATE_SIDE_MIN) or right_count < int(POSE_GATE_SIDE_MIN):
            return False

        return True
    
     # ========== TTAユーティリティ ==========
    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v if n < eps else (v / n)

    @staticmethod
    def _reverse_sequence_xy(seq: np.ndarray) -> np.ndarray:
        """(T,K,2) を時間反転"""
        a = np.asarray(seq, dtype=np.float32)
        return a[::-1].copy()

    @staticmethod
    def _flip_sequence_xy(seq: np.ndarray) -> np.ndarray:
        """
        (T,K,2) の左右反転。xのみ符号反転（正規化座標前提）。左右ジョイント入替が必要なら
        ここに既存の左右ペアスワップ処理を差し込むか、あなたの既存関数で置き換えてください。
        """
        a = np.asarray(seq, dtype=np.float32).copy()
        a[..., 0] = -a[..., 0]
        # 例：左右ペアのスワップが必要ならここで実施（あなたの既存スワップ実装があれば置換）
        return a

    def _make_tta_sequences(self, seq_xy: np.ndarray) -> List[np.ndarray]:
        """config.TTA_MODES の順に (T,K,2) シーケンス群を生成"""
        out = []
        for m in TTA_MODES:
            if m == "normal":
                out.append(seq_xy)
            elif m == "flip":
                out.append(self._flip_sequence_xy(seq_xy))
            elif m == "rev":
                out.append(self._reverse_sequence_xy(seq_xy))
            elif m == "rev_flip":
                out.append(self._flip_sequence_xy(self._reverse_sequence_xy(seq_xy)))
            else:
                raise ValueError(f"Unknown TTA mode: {m}")
        return out

    # ========== 埋め込み生成 ==========
    def _embed_tta_unit(self, seq_xy: np.ndarray) -> List[np.ndarray]:
        """
        各TTAで 128D を生成（ここでは正規化しない）
        前提：self.embed_sequence(seq_xy) が 128D を返す既存実装
        """
        embs: List[np.ndarray] = []
        for s in self._make_tta_sequences(seq_xy):
            e = self.embed_sequence(s)  # 既存：1本の (T,K,2) → 128D
            if e is None:
                e = np.zeros((128,), dtype=np.float32)
            embs.append(e.astype(np.float32, copy=False))
        return embs  # [128D, 128D, ...]

    def _embed_tta_concat(self, seq_xy: np.ndarray) -> np.ndarray:
        """128D×本数を結合 → 結合後に L2 正規化を1回だけ"""
        unit = self._embed_tta_unit(seq_xy)              # list of 128D
        concat = np.concatenate(unit, axis=-1).astype(np.float32, copy=False)
        return self._l2_normalize(concat)

    # ========== 類似度（安定軸：TopK平均 / 伸び軸：concat） ==========
    def _cos_topk_mean(self, gal_units: List[np.ndarray], qry_units: List[np.ndarray], k: int) -> float:
        """TTAごとのcosを並べ、上位K本の平均"""
        if not gal_units or not qry_units:
            return 0.0
        sims = []
        for ge, qe in zip(gal_units, qry_units):
            gn = self._l2_normalize(ge)
            qn = self._l2_normalize(qe)
            sims.append(float(np.dot(gn, qn)))
        sims.sort(reverse=True)
        kk = max(1, min(k, len(sims)))
        return float(np.mean(sims[:kk]))

    def _cos_concat(self, gal_concat: np.ndarray, qry_concat: np.ndarray) -> float:
        gn = self._l2_normalize(gal_concat)
        qn = self._l2_normalize(qry_concat)
        return float(np.dot(gn, qn))

    # ========== 公開API：ハイブリッド最終スコア ==========
    def hybrid_similarity(
        self,
        gallery_seq_xy: np.ndarray,
        query_seq_xy: np.ndarray,
        *,
        gallery_cache: Optional[Tuple[List[np.ndarray], np.ndarray]] = None,
        alpha: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Tuple[float, dict]:
        """
        最終スコア = α * concat_cos + (1-α) * TopK平均
        - gallery_cache: (gal_units[128D×本数], gal_concat[128*本数]) を渡すと高速化
        - 返り値: (score, debug_info)
        """
        if alpha is None:
            alpha = HYBRID_TTA_ALPHA
        if topk is None:
            topk = TTA_SCORE_TOPK

        # ギャラリー側（キャッシュ優先）
        if gallery_cache is not None:
            gal_units, gal_concat = gallery_cache
        else:
            gal_units = self._embed_tta_unit(gallery_seq_xy)
            gal_concat = self._embed_tta_concat(gallery_seq_xy)

        # クエリ側
        qry_units = self._embed_tta_unit(query_seq_xy)
        qry_concat = self._embed_tta_concat(query_seq_xy)

        # 2軸スコア
        s_topk = self._cos_topk_mean(gal_units, qry_units, k=topk)
        s_concat = self._cos_concat(gal_concat, qry_concat)

        # HYBRID無効 or α=0 → TopKのみ
        score = s_topk if (not HYBRID_TTA_ENABLED or alpha <= 0.0) else (alpha * s_concat + (1.0 - alpha) * s_topk)

        dbg = {
            "alpha": alpha,
            "topk": topk,
            "tta_modes": list(TTA_MODES),
            "s_topk": s_topk,
            "s_concat": s_concat,
            "score": score,
        }
        return score, dbg

    def build_gallery_tta_cache(self, gallery_seq_xy: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        ギャラリー用の TTA埋め込み（128D×本数）と concat埋め込み（128*本数）を返す。
        build_gallery_from_buffer() の最後で self._gallery_tta_cache に格納するのが推奨。
        """
        units = self._embed_tta_unit(gallery_seq_xy)
        concat = self._embed_tta_concat(gallery_seq_xy)
        return units, concat
    
    # ========== 損失関数用ユーティリティ ==========
    @staticmethod
    def apply_temperature_softmax(similarities: np.ndarray, tau: float = 0.01) -> np.ndarray:
        """
        類似度ベクトルに温度スケーリングを適用してsoftmax確率に変換する。
        SupCon Lossなどと同じ形式で、tauが小さいほど鋭く区別される。
        Args:
            similarities: np.ndarray shape=(R,) 類似度ベクトル
            tau: 温度係数（小さいほどsharp）
        Returns:
            np.ndarray shape=(R,) softmax確率分布（合計1.0）
        """
        if similarities is None or len(similarities) == 0:
            return np.array([], dtype=np.float32)
        tau = max(float(tau), 1e-6)
        x = np.asarray(similarities, dtype=np.float32) / tau
        x = x - np.max(x)  # 数値安定化
        e = np.exp(x)
        p = e / (np.sum(e) + 1e-12)
        return p.astype(np.float32)
    
    # ========== 類似度後処理（0..1へ拡張） ==========
    @staticmethod
    def score_calibrate_vector(
        sims: np.ndarray,
        kind: str = "fisher",
        *,
        beta: float = 0.95,
        eps: float = 1e-6
    ) -> np.ndarray:
        """
        cos類似度ベクトル(sims)を 0..1 へ写像して“団子”をほぐす。
        kind:
          - "fisher":  y = (atanh(beta*s) / atanh(beta) + 1)/2
          - "rank"  :  y = rank(s) / (R-1)          （同一ベクトル内の順位で0..1）
          - "minmax":  y = (s - min)/(max - min)     （同一ベクトル内の線形）
          - "none"  :  変換しない（simsを返す）
        返り値は shape=(R,) の float32
        """
        if sims is None:
            return np.array([], dtype=np.float32)
        x = np.asarray(sims, dtype=np.float32).copy()
        if x.size == 0:
            return x

        kind = (kind or "none").lower()
        if kind == "none":
            return x

        if kind == "fisher":
            # cos∈[-1,1] を安全に処理
            b = float(beta)
            e = float(eps)
            b = np.clip(b, 0.0 + 1e-6, 1.0 - 1e-6)
            xe = np.clip(b * x, -1.0 + e, 1.0 - e)
            num = np.arctanh(xe)
            den = np.arctanh(np.clip(b, -1.0 + e, 1.0 - e))
            y = num / (den + e)           # 正規化：[-1,1] 付近へ
            y = (y + 1.0) * 0.5           # [0,1] へ
            return y.astype(np.float32)

        if kind == "rank":
            # 大きいほど1に近い値（ tiesは平均順位にしてもOKだが簡素化）
            order = np.argsort(x)              # 昇順
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(x.size, dtype=np.float32)
            denom = max(1.0, x.size - 1.0)
            y = ranks / denom                  # [0,1]
            return y

        if kind == "minmax":
            mn = float(np.min(x)); mx = float(np.max(x))
            if mx - mn < 1e-12:
                return np.full_like(x, 0.5, dtype=np.float32)
            y = (x - mn) / (mx - mn)
            return y.astype(np.float32)

        # 未知指定は素通し
        return x
