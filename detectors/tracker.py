# -*- coding: utf-8 -*-
"""
detectors/tracker.py - 超簡易 ByteTrack 風トラッカー
- 入力: dets (N,5) [x1,y1,x2,y2,score]
- 出力: Track オブジェクトのリスト（track_id, tlbr などを保持）
- KalmanFilter は使わず、IoU + スコア2段階マッチングのみ実装
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np


class Track:
    """単一オブジェクトのトラック状態（超簡易版）"""
    _next_id = 1

    def __init__(self, tlbr: np.ndarray, score: float):
        # tlbr = [x1,y1,x2,y2]
        self.tlbr = tlbr.astype(np.float32)
        self.score = float(score)
        self.track_id = Track._next_id
        Track._next_id += 1

        self.age = 1                  # 生存フレーム数
        self.time_since_update = 0    # 最後に更新されてからのフレーム数
        self._updated_this_frame = False

    def mark_missed(self) -> None:
        """このフレームで更新されなかったトラックを「見失い」状態として進める"""
        if not self._updated_this_frame:
            self.time_since_update += 1
        self._updated_this_frame = False

    def update(self, tlbr: np.ndarray, score: float) -> None:
        """検出結果でトラックを更新"""
        self.tlbr = tlbr.astype(np.float32)
        self.score = float(score)
        self.age += 1
        self.time_since_update = 0
        self._updated_this_frame = True


def iou_matrix(tracks: List[Track], dets_tlbr: np.ndarray) -> np.ndarray:
    """
    tracks と dets_tlbr の IoU 行列を計算
    tracks: List[Track]
    dets_tlbr: (D,4) [x1,y1,x2,y2]
    戻り値: (T,D) の IoU 行列
    """
    if len(tracks) == 0 or dets_tlbr.size == 0:
        return np.zeros((len(tracks), len(dets_tlbr)), dtype=np.float32)

    t_boxes = np.stack([t.tlbr for t in tracks], axis=0)  # (T,4)
    d_boxes = dets_tlbr.astype(np.float32)                # (D,4)

    t_x1, t_y1, t_x2, t_y2 = np.split(t_boxes, 4, axis=1)
    d_x1, d_y1, d_x2, d_y2 = np.split(d_boxes, 4, axis=1)

    inter_x1 = np.maximum(t_x1, d_x1.T)
    inter_y1 = np.maximum(t_y1, d_y1.T)
    inter_x2 = np.minimum(t_x2, d_x2.T)
    inter_y2 = np.minimum(t_y2, d_y2.T)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
    area_d = (d_x2 - d_x1) * (d_y2 - d_y1)
    union = area_t + area_d.T - inter
    iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return iou


class ByteTracker:
    """
    超簡易ByteTrack風トラッカー
    - Kalman Filter なし
    - IoU + スコア2段階マッチングのみ実装
    - 入力: dets (N,5) [x1,y1,x2,y2,score]
    - 出力: List[Track]
    """
    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.6,
        match_thresh: float = 0.7,
        max_age: int = 30,
    ) -> None:
        self.track_thresh = float(track_thresh)
        self.high_thresh = float(high_thresh)
        self.match_thresh = float(match_thresh)
        self.max_age = int(max_age)

        self.tracks: List[Track] = []

    def _greedy_match(self, tracks: List[Track], dets_tlbr: np.ndarray, match_thresh: float):
        """
        IoU 行列から貪欲にマッチングする（Hungarian ではなく簡易版）
        戻り値: (matched, unmatched_tracks, unmatched_dets)
        """
        if len(tracks) == 0 or dets_tlbr.size == 0:
            return [], list(range(len(tracks))), list(range(len(dets_tlbr)))

        iou = iou_matrix(tracks, dets_tlbr)  # (T,D)
        matched = []
        used_t = set()
        used_d = set()

        while True:
            t_idx, d_idx = divmod(int(iou.argmax()), iou.shape[1])
            if iou[t_idx, d_idx] < match_thresh:
                break
            if t_idx in used_t or d_idx in used_d:
                iou[t_idx, d_idx] = -1.0
                continue
            matched.append((t_idx, d_idx))
            used_t.add(t_idx)
            used_d.add(d_idx)
            iou[t_idx, :] = -1.0
            iou[:, d_idx] = -1.0

        unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(dets_tlbr)) if i not in used_d]
        return matched, unmatched_t, unmatched_d

    def update(self, dets: Optional[np.ndarray]) -> List[Track]:
        """
        dets: (N,5) [x1,y1,x2,y2,score] or None
        戻り値: 現在アクティブなTrackのリスト（time_since_update <= max_age）
        """
        # 1) 全トラックを「未更新」にしておく
        for t in self.tracks:
            t._updated_this_frame = False

        if dets is None or len(dets) == 0:
            # 検出なし → time_since_updateだけ進める
            for t in self.tracks:
                t.mark_missed()
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return list(self.tracks)

        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim != 2 or dets.shape[1] != 5:
            raise ValueError("dets must be (N,5) [x1,y1,x2,y2,score]")

        # スコアで2段階に分ける
        scores = dets[:, 4]
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.track_thresh) & (~high_mask)

        dets_high = dets[high_mask]
        dets_low = dets[low_mask]

        # 2) 高スコア検出とのマッチング
        matched, u_tracks, u_high = self._greedy_match(self.tracks, dets_high[:, :4], self.match_thresh)

        for t_idx, d_idx in matched:
            tlbr = dets_high[d_idx, :4]
            score = float(dets_high[d_idx, 4])
            self.tracks[t_idx].update(tlbr, score)

        # 3) 残ったトラックと低スコア検出で再マッチング
        remain_tracks = [self.tracks[i] for i in u_tracks]
        matched2, u_tracks2_local, u_low = self._greedy_match(remain_tracks, dets_low[:, :4], self.match_thresh)

        for local_t_idx, d_idx in matched2:
            global_t_idx = u_tracks[local_t_idx]
            tlbr = dets_low[d_idx, :4]
            score = float(dets_low[d_idx, 4])
            self.tracks[global_t_idx].update(tlbr, score)

        # 4) 未マッチのトラックは time_since_update++
        updated_ids = {t.track_id for t in self.tracks if t._updated_this_frame}
        for t in self.tracks:
            if t.track_id not in updated_ids:
                t.mark_missed()

        # 5) 未マッチの高スコア検出から新規トラックを生成
        high_indices = np.where(high_mask)[0]
        new_indices = [high_indices[i] for i in u_high]
        for det_idx in new_indices:
            tlbr = dets[det_idx, :4]
            score = float(dets[det_idx, 4])
            self.tracks.append(Track(tlbr, score))

        # 6) 古いトラックを削除
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return list(self.tracks)
