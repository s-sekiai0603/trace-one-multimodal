# -*- coding: utf-8 -*-
"""
auto_ranker.py — マルチモーダルモード用ランキングロジック
- BBOXごとの Face/App/Gait 類似度から、上位N件を決める
- ランキングモード "face" :
    顔の類似度が取れているBBOXが1つでもあれば Face だけでランキング
    → 1件もなければ Appearance
    → それも全滅なら Gait
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import math
import numpy as np


@dataclass
class RankEntry:
    """ランキング結果1件分"""
    index: int                 # 対応するBBOXインデックス
    rank: int                  # 1〜N位
    used_modality: str         # "face" / "app" / "gait"
    face_sim: Optional[float]  # そのBBOXのFace類似度（あれば）
    app_sim: Optional[float]   # 外見類似度
    gait_sim: Optional[float]  # 歩容類似度


class AutoRanker:
    """
    Autoモード用ランキングクラス。
    - mode="face" のみ実装（他モードは将来拡張用）
    """

    def __init__(self, mode: str = "face") -> None:
        self.mode = mode

    # ---- 公開API ---------------------------------------------------------

    def rank_bboxes(
        self,
        sim_face: Optional[Sequence[float]],
        sim_app: Optional[Sequence[float]],
        sim_gait: Optional[Sequence[float]],
        num_boxes: int,
        topn: int,
    ) -> List[RankEntry]:
        """
        BBOXごとの類似度配列から、上位 topn 件を返す。

        Parameters
        ----------
        sim_face, sim_app, sim_gait :
            長さ num_boxes の配列（またはそれ以下）。None や NaN は「スコアなし」とみなす。
        num_boxes :
            Pose BBOX の個数（len(self._boxes_pose)）
        topn :
            何位まで返すか

        Returns
        -------
        List[RankEntry]
        """
        n = int(num_boxes)
        if n <= 0:
            return []

        face = self._normalize_sim_array(sim_face, n)
        app  = self._normalize_sim_array(sim_app, n)
        gait = self._normalize_sim_array(sim_gait, n)

        # どのモダリティでランキングするか決定
        used_label, arr_used = self._select_modality(face, app, gait)
        if used_label is None or arr_used is None:
            return []

        # スコアが存在するインデックスだけ集めてソート
        scored: List[tuple[int, float]] = []
        for i in range(n):
            v = arr_used[i]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            scored.append((i, float(v)))

        if not scored:
            return []

        scored.sort(key=lambda t: t[1], reverse=True)
        scored = scored[:max(1, int(topn))]

        entries: List[RankEntry] = []
        for rank, (idx, _) in enumerate(scored, start=1):
            entries.append(
                RankEntry(
                    index=idx,
                    rank=rank,
                    used_modality=used_label,
                    face_sim=face[idx] if face is not None else None,
                    app_sim=app[idx] if app is not None else None,
                    gait_sim=gait[idx] if gait is not None else None,
                )
            )
        return entries

    # ---- 内部ユーティリティ ----------------------------------------------

    @staticmethod
    def _normalize_sim_array(
        arr: Optional[Sequence[float]],
        n: int,
    ) -> Optional[list[Optional[float]]]:
        """
        類似度配列を長さ n の list[Optional[float]] にそろえる。
        - None：スコアなし
        - NaN：スコアなしとして扱う
        """
        if arr is None:
            return None

        # numpy配列なども list にして扱う
        if isinstance(arr, np.ndarray):
            arr_list = arr.tolist()
        else:
            arr_list = list(arr)

        # 長さを調整
        if len(arr_list) < n:
            arr_list = arr_list + [None] * (n - len(arr_list))
        elif len(arr_list) > n:
            arr_list = arr_list[:n]

        # float にキャスト（None はそのまま）
        out: list[Optional[float]] = []
        for v in arr_list:
            if v is None:
                out.append(None)
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(None)
        return out

    def _select_modality(
        self,
        face: Optional[Sequence[Optional[float]]],
        app: Optional[Sequence[Optional[float]]],
        gait: Optional[Sequence[Optional[float]]],
    ) -> tuple[Optional[str], Optional[Sequence[Optional[float]]]]:
        """
        ランキングに使うモダリティと、その配列を返す。
        mode="face" の場合:
          - 顔に1件でも有効スコアがあれば face
          - ダメなら app
          - それも全滅なら gait
        """
        if self.mode == "face":
            if self._has_valid_score(face):
                return "face", face
            if self._has_valid_score(app):
                return "app", app
            if self._has_valid_score(gait):
                return "gait", gait
            return None, None

        # 将来モード追加用：とりあえず face/app/gait 順
        for label, arr in (("face", face), ("app", app), ("gait", gait)):
            if self._has_valid_score(arr):
                return label, arr
        return None, None

    @staticmethod
    def _has_valid_score(arr: Optional[Sequence[Optional[float]]]) -> bool:
        if arr is None:
            return False
        for v in arr:
            if v is None:
                continue
            try:
                f = float(v)
            except Exception:
                continue
            if not math.isnan(f):
                return True
        return False
