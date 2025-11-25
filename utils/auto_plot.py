# -*- coding: utf-8 -*-
"""
auto_plot.py — マルチモーダル識別時の TrackID別類似度プロット管理

- 顔 / 外見 / 歩容 の 3種類について、track_id ごとの類似度推移を折れ線グラフで可視化
- 顔: 緑、外見: 青、歩容: 紫
- クリックした線のみ赤色にして強調（対象 track_id の切り替え）
- 歩容のみ、識別開始から AUTO_GAIT_FRAMES 経過後に描画開始
- AUTO_TRACK_RANK_INTERVAL_FRAMES フレームごとに
    - グラフを PNG 出力
    - グラフを一度終了（内部状態リセット）して再開
  するインターバルモードに対応（0 の場合は識別開始〜終了で1本）

使い方イメージ（AutoController 側）:
    from .auto_plot import AutoPlotManager

    self._plot_mgr = AutoPlotManager(
        enabled=True,
        video_time_label="00:02",  # 動画再生位置などを渡す
    )

    # 識別ONになった瞬間
    self._plot_mgr.start_session()

    # 毎フレームの識別結果更新時:
    self._plot_mgr.update(
        frame_idx=self._capture_frame_idx,
        face_scores_by_tid=face_scores_dict,
        app_scores_by_tid=app_scores_dict,
        gait_scores_by_tid=gait_scores_dict,
    )

    # 識別OFFになった瞬間
    self._plot_mgr.finalize()

"""

from __future__ import annotations
from typing import Dict, Optional
import os
from datetime import datetime

import numpy as np

# config から設定値を取る（パッケージ / 単体実行どちらでも動くように二段構え）
try:
    from ..config import AUTO_TRACK_RANK_INTERVAL_FRAMES, AUTO_GAIT_FRAMES
except Exception:  # 単体で動かす場合など
    try:
        from config import AUTO_TRACK_RANK_INTERVAL_FRAMES, AUTO_GAIT_FRAMES
    except Exception:
        AUTO_TRACK_RANK_INTERVAL_FRAMES = 0
        AUTO_GAIT_FRAMES = 32


# ================================================================
# 基本プロッタ（SimilarityPlotter をベースに拡張）
# ================================================================
class SimilarityPlotter:
    """
    識別中の「全track_idの類似度」をフレーム軸に沿って折れ線で可視化する。

    - update(frame_idx, {tid: score or np.nan}) を毎フレーム呼ぶだけ
    - target_tid を指定すると、その track_id の線だけ赤色で強調
    - 線をクリックすると、その線の track_id が target になる
    """

    def __init__(
        self,
        enabled: bool = True,
        max_points: int = 600,
        title: str = "Similarity / track_id",
        base_color: str = "blue",
        target_color: str = "red",
    ):
        self.enabled = bool(enabled)
        self.max_points = int(max_points)
        self.title = str(title)
        self.base_color = str(base_color)
        self.target_color = str(target_color)

        self._fig = None
        self._ax = None
        self._lines: Dict[int, any] = {}       # tid -> Line2D
        self._line_to_tid: Dict[any, int] = {} # Line2D -> tid
        self._xs: Dict[int, list] = {}
        self._ys: Dict[int, list] = {}
        self._target_tid: Optional[int] = None
        self._last_frame_idx: int = -1
        self._legend_drawn: bool = False
        self._cid_pick: Optional[int] = None   # mpl_connect handle

    # ---------- lifecycle ----------
    def show(self):
        """
        プロットの初期化。
        - 図サイズは正方形に固定
        - Axes のボックス比率も 1:1 に固定
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter

        if not self.enabled:
            return

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

            # 図サイズ（好みで調整）
            try:
                self._fig.set_size_inches(6, 6, forward=True)
            except Exception:
                pass

            # 軸のボックスを常に正方形に
            try:
                self._ax.set_box_aspect(1)
            except Exception:
                try:
                    self._ax.set_aspect("equal", adjustable="box")
                except Exception:
                    pass

            self._ax.set_title(self.title)
            self._ax.set_xlabel("Frame index")
            self._ax.set_ylabel("Similarity (cos)")

            self._ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
            self._ax.set_autoscaley_on(False)
            self._ax.set_xlim(0, 1)
            self._ax.set_autoscalex_on(False)
            self._ax.grid(True, alpha=0.3)

            self._legend_drawn = False

            # 線クリック用イベント
            try:
                self._cid_pick = self._fig.canvas.mpl_connect(
                    "pick_event", self._on_pick
                )
            except Exception:
                self._cid_pick = None

        # 内部状態の初期化
        if not hasattr(self, "_lines"): self._lines = {}
        if not hasattr(self, "_line_to_tid"): self._line_to_tid = {}
        if not hasattr(self, "_xs"): self._xs = {}
        if not hasattr(self, "_ys"): self._ys = {}
        if not hasattr(self, "max_points"): self.max_points = 512
        if not hasattr(self, "_target_tid"): self._target_tid = None
        if not hasattr(self, "_last_frame_idx"): self._last_frame_idx = 0

    def close(self):
        import matplotlib.pyplot as plt

        if self._fig is not None:
            try:
                if self._cid_pick is not None:
                    try:
                        self._fig.canvas.mpl_disconnect(self._cid_pick)
                    except Exception:
                        pass
            except Exception:
                pass

            plt.close(self._fig)

        self._fig = None
        self._ax = None
        self._lines.clear()
        self._line_to_tid.clear()
        self._xs.clear()
        self._ys.clear()
        self._target_tid = None
        self._last_frame_idx = -1
        self._legend_drawn = False
        self._cid_pick = None

    def reset(self):
        """
        グラフを一度閉じて、空の状態で再作成。
        """
        self.close()
        if self.enabled:
            self.show()

    # ---------- control ----------
    def set_target_tid(self, tid: Optional[int]):
        """赤線にする対象の track_id を切替（Noneで解除）"""
        self._target_tid = tid
        # 既存ラインの色を塗り直し
        for k, ln in self._lines.items():
            if ln is None:
                continue
            if self._target_tid is not None and k == self._target_tid:
                ln.set_color(self.target_color); ln.set_linewidth(2.5); ln.set_alpha(1.0)
            else:
                ln.set_color(self.base_color);   ln.set_linewidth(1.2); ln.set_alpha(0.9)
        if self._fig is not None:
            try:
                self._fig.canvas.draw_idle()
            except Exception:
                pass

    # ---------- event handlers ----------
    def _on_pick(self, event):
        """
        Matplotlib の pick_event から呼ばれる。
        クリックされた Line2D に対応する tid を target にする。
        """
        line = getattr(event, "artist", None)
        if line is None:
            return
        tid = self._line_to_tid.get(line, None)
        if tid is None:
            # 逆引き（保険）
            for k, ln in self._lines.items():
                if ln is line:
                    tid = k
                    break
        if tid is None:
            return
        try:
            tid = int(tid)
        except Exception:
            pass
        self.set_target_tid(tid)

    # ---------- main update ----------
    def update(self, frame_idx: int, scores_by_tid, target_tid=None):
        """
        1フレームぶんのスコアを投入して描画を更新する。

        - frame_idx: 1,2,3,... と増えるフレーム番号（0始まりなら +1 して渡す）
        - scores_by_tid: { tid(int): score(float or None) }
        - target_tid: 強調表示したい tid（None なら維持）
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter

        # 描画有効チェック
        if not self.enabled:
            return

        # 図の用意
        if self._fig is None or self._ax is None:
            self.show()
            if self._fig is None or self._ax is None:
                return

        # 内部状態の初期化（未定義なら）
        if not hasattr(self, "_lines"): self._lines = {}
        if not hasattr(self, "_line_to_tid"): self._line_to_tid = {}
        if not hasattr(self, "_xs"): self._xs = {}
        if not hasattr(self, "_ys"): self._ys = {}
        if not hasattr(self, "max_points"): self.max_points = 512
        if not hasattr(self, "_target_tid"): self._target_tid = None
        if not hasattr(self, "_last_frame_idx"): self._last_frame_idx = 0

        # ターゲット更新
        if target_tid is not None:
            try:
                self._target_tid = int(target_tid)
            except Exception:
                self._target_tid = target_tid  # 文字列でも動くように

        # 受け取り整形
        scores_by_tid = scores_by_tid or {}
        present_tids = set()

        # それぞれの tid にデータを追加
        for tid, sc in scores_by_tid.items():
            try:
                tid = int(tid)
            except Exception:
                pass
            present_tids.add(tid)

            if tid not in self._ys:
                self._ys[tid] = []
                self._xs[tid] = []
                # 線を作成（クリック検出のため picker をON）
                line, = self._ax.plot(
                    [], [],
                    color=self.base_color,
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"tid={tid}",
                    picker=5.0,     # ピクセルでの許容距離
                )
                self._lines[tid] = line
                self._line_to_tid[line] = tid

            # 値を追加
            self._xs[tid].append(int(frame_idx))
            self._ys[tid].append(np.nan if sc is None else float(sc))

            # 上限超えたら古い点を捨てる（スクロール）
            if len(self._ys[tid]) > self.max_points:
                self._ys[tid] = self._ys[tid][-self.max_points:]
                self._xs[tid] = self._xs[tid][-self.max_points:]

        # データ反映＆色分け
        for tid, line in self._lines.items():
            xs = self._xs.get(tid, [])
            ys = self._ys.get(tid, [])
            line.set_data(xs, ys)

            if self._target_tid is not None and tid == self._target_tid:
                line.set_color(self.target_color); line.set_linewidth(2.4); line.set_alpha(1.0)
            else:
                line.set_color(self.base_color);   line.set_linewidth(1.2); line.set_alpha(0.9)

        # 軸：Xは明示、Yは観測値に応じて動的
        xmin = max(1, int(frame_idx) - self.max_points + 1)
        xmax = max(1, int(frame_idx))
        self._ax.set_xlim(xmin, xmax)

        # ==== 動的Yスケーリング ====
        vals = []
        for ys in self._ys.values():
            if not ys:
                continue
            take = ys[-self.max_points:]
            for v in take:
                if v is None:
                    continue
                if isinstance(v, (float, int)) and v == v:  # NaN除外
                    vals.append(float(v))

        if vals:
            y_min = min(vals)
            y_max = max(vals)

            if abs(y_max - y_min) < 1e-6:
                y_min -= 0.0005
                y_max += 0.0005

            rel_margin = 0.05
            min_margin = 0.001
            margin = max(min_margin, (y_max - y_min) * rel_margin)

            y0 = max(0.0, y_min - margin)
            y1 = min(1.0, y_max + margin)
        else:
            y0, y1 = 0.8, 1.0

        self._ax.set_ylim(y0, y1)
        # ==========================

        self._ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        self._ax.grid(True, alpha=0.3)

        # 凡例
        if not self._legend_drawn:
            try:
                self._ax.legend(loc="lower right", fontsize=8)
            except Exception:
                pass
            self._legend_drawn = True

        # 正方形維持
        try:
            self._ax.set_box_aspect(1)
        except Exception:
            try:
                self._ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass

        try:
            self._fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass

        self._last_frame_idx = int(frame_idx)

    # ---------- 保存 ----------
    def save(self, filepath: str) -> None:
        """
        現在のグラフを PNG などに保存する。
        """
        if not self.enabled or self._fig is None:
            return
        import matplotlib.pyplot as plt

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except Exception:
            pass

        try:
            self._fig.savefig(filepath, dpi=150, bbox_inches="tight")
        except Exception:
            pass


# ================================================================
# 3モダリティ統合マネージャ
# ================================================================
class AutoPlotManager:
    """
    顔 / 外見 / 歩容 の3種類の SimilarityPlotter をまとめて管理するクラス。

    - 顔: 緑線
    - 外見: 青線
    - 歩容: 紫線（AUTO_GAIT_FRAMES 経過後から描画）
    - AUTO_TRACK_RANK_INTERVAL_FRAMES ごとにグラフを保存＆リセット
    """

    def __init__(
        self,
        enabled: bool = True,
        root_dir: str = "data/evidence/plot",
        interval_frames: Optional[int] = None,
        gait_start_frames: Optional[int] = None,
        video_time_label: str = "00:00",
    ):
        self.enabled = bool(enabled)
        self.root_dir = root_dir

        if interval_frames is None:
            self.interval_frames = int(AUTO_TRACK_RANK_INTERVAL_FRAMES)
        else:
            self.interval_frames = int(interval_frames)

        if gait_start_frames is None:
            self.gait_start_frames = int(AUTO_GAIT_FRAMES)
        else:
            self.gait_start_frames = int(gait_start_frames)

        # 動画の開始時間ラベル（例 "00:02"）
        self.video_time_label = str(video_time_label)

        # セッションディレクトリ（data/evidence/plot/1125_153045 など）
        self._session_dir: Optional[str] = None

        # プロッタ
        self._plot_face: Optional[SimilarityPlotter] = None
        self._plot_app: Optional[SimilarityPlotter] = None
        self._plot_gait: Optional[SimilarityPlotter] = None

        # カウンタ系
        self._frame_counter: int = 0              # 識別ON中の累積フレーム数
        self._segment_start_frame: int = 0        # 現セグメントの開始フレーム
        self._segment_index: int = 0              # 何本目のグラフか

        # gait 描画開始フラグ
        self._gait_started: bool = False

    # ---------- lifecycle ----------
    def start_session(self) -> None:
        """
        新しい識別セッションを開始。
        - 出力先フォルダを data/evidence/plot/<月日_時分秒>/ に作成
        - 3つのプロッタを初期化して表示
        """
        if not self.enabled:
            return

        # セッションフォルダ作成（例: data/evidence/plot/1125_153045）
        now = datetime.now()
        ts = now.strftime("%m%d_%H%M%S")
        self._session_dir = os.path.join(self.root_dir, ts)
        os.makedirs(self._session_dir, exist_ok=True)

        # --- プロッタの生成 or リセット ---
        if self._plot_face is None:
            self._plot_face = SimilarityPlotter(
                enabled=True,
                max_points=600,
                title="Face similarity (per track_id)",
                base_color="green",
                target_color="red",
            )
            self._plot_face.show()
        else:
            # 既存のウィンドウを閉じて中身だけリセット
            self._plot_face.reset()

        if self._plot_app is None:
            self._plot_app = SimilarityPlotter(
                enabled=True,
                max_points=600,
                title="Appearance similarity (per track_id)",
                base_color="blue",
                target_color="red",
            )
            self._plot_app.show()
        else:
            self._plot_app.reset()

        if self._plot_gait is None:
            self._plot_gait = SimilarityPlotter(
                enabled=True,
                max_points=600,
                title="Gait similarity (per track_id)",
                base_color="purple",
                target_color="red",
            )
            self._plot_gait.show()
        else:
            self._plot_gait.reset()

        # カウンタリセット
        self._frame_counter = 0
        self._segment_start_frame = 0
        self._segment_index = 0
        self._gait_started = False


    def finalize(self) -> None:
        """
        セッション終了時に呼ぶ。
        - 未保存のグラフがあれば保存して終了。
        """
        if not self.enabled:
            return

        # まだ1フレームも処理していないなら何もしない
        if self._frame_counter <= 0:
            return

        # 最後のセグメントを保存
        self._segment_index += 1
        self._save_all_plots()

    # ---------- main update ----------
    def update(
        self,
        frame_idx: int,
        face_scores_by_tid: Optional[Dict[int, float]] = None,
        app_scores_by_tid: Optional[Dict[int, float]] = None,
        gait_scores_by_tid: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        毎フレーム呼び出すメソッド。

        - frame_idx: 識別ON中のフレーム番号（1始まり推奨）
        - *_scores_by_tid: {track_id: similarity or None}
        """
        if not self.enabled:
            return
        if self._session_dir is None:
            # start_session() が呼ばれていない場合は何もしない
            return

        self._frame_counter = int(frame_idx)

        # プロッタが未初期化なら念のため作る（start_session() が通常はやる）
        if self._plot_face is None or self._plot_app is None or self._plot_gait is None:
            self.start_session()

        # 顔 / 外見 は1フレーム目から更新
        if self._plot_face is not None:
            self._plot_face.update(frame_idx, face_scores_by_tid or {})
        if self._plot_app is not None:
            self._plot_app.update(frame_idx, app_scores_by_tid or {})

        # 歩容は「識別開始から gait_start_frames フレーム経過」後に開始
        if frame_idx >= self.gait_start_frames:
            self._gait_started = True
        if self._gait_started and self._plot_gait is not None:
            self._plot_gait.update(frame_idx, gait_scores_by_tid or {})

        # インターバル管理
        if self.interval_frames > 0:
            # 現セグメント内での経過フレーム数
            seg_len = self._frame_counter - self._segment_start_frame
            if seg_len >= self.interval_frames:
                # セグメントを締めて保存 → リセット → 次のセグメント開始
                self._segment_index += 1
                self._save_all_plots()
                self._reset_plots_for_next_segment()
                self._segment_start_frame = self._frame_counter

    # ---------- internal helpers ----------
    def _safe_video_label(self) -> str:
        """
        ファイル名用に動画開始時間ラベルから ':' を '-' に変換。
        """
        label = self.video_time_label or "start"
        return label.replace(":", "-")

    def _save_all_plots(self) -> None:
        """
        現在の3つのグラフを PNG に保存する。
        """
        if self._session_dir is None:
            return

        safe_label = self._safe_video_label()

        # 複数セグメントがある場合は _partNN をつける（1本目は省略）
        part_suffix = ""
        if self._segment_index > 1:
            part_suffix = f"_part{self._segment_index:02d}"

        face_name = f"face_plot_{safe_label}{part_suffix}.png"
        app_name  = f"app_plot_{safe_label}{part_suffix}.png"
        gait_name = f"gait_plot_{safe_label}{part_suffix}.png"

        face_path = os.path.join(self._session_dir, face_name)
        app_path  = os.path.join(self._session_dir, app_name)
        gait_path = os.path.join(self._session_dir, gait_name)

        if self._plot_face is not None:
            self._plot_face.save(face_path)
        if self._plot_app is not None:
            self._plot_app.save(app_path)
        if self._plot_gait is not None:
            self._plot_gait.save(gait_path)

    def _reset_plots_for_next_segment(self) -> None:
        """
        セグメント境界でグラフをリセットして、次のセグメント用に空にする。
        """
        if self._plot_face is not None:
            self._plot_face.reset()
        if self._plot_app is not None:
            self._plot_app.reset()
        if self._plot_gait is not None:
            self._plot_gait.reset()

        # gait の開始判定は全体の frame_idx に対して行うのでフラグは保持
        # （識別開始から Nフレーム経過したら以降ずっと True のまま）
