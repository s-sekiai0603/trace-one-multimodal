# -*- coding: utf-8 -*-
"""
auto_report_html.py — マルチモーダルモード HTMLレポート生成
- 時刻ごと（例: 04:12, 04:14, ...）に
  上位N件の人物キャプチャと類似度をHTMLに整形して出力する
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import os
import math
import html
import base64


@dataclass
class ReportItem:
    """1つのランキング枠（1位,2位,...）に対応する情報"""
    rank: int
    image_path: str          # 実ファイルパス（<img src> では basename にする）
    face_sim: Optional[float]
    app_sim: Optional[float]
    gait_sim: Optional[float]


@dataclass
class ReportSnapshot:
    """
    1回分のランキングスナップショット
    例: time_label="04:12" に対して、1〜N位の ReportItem を持つ
    """
    time_label: str
    items: List[ReportItem]


class AutoHtmlReportBuilder:
    """
    マルチモーダルモード HTMLレポートビルダー。

    使い方（auto_controller側の想定）:
      builder = AutoHtmlReportBuilder(output_dir=..., show_sim=True)
      builder.set_video_name("xxx.mp4")
      builder.build_from_snapshots(snapshots, basename="auto_report.html")
    """

    def __init__(self, output_dir: str, show_sim: bool = True) -> None:
        self.output_dir = output_dir
        self.show_sim = show_sim
        self.video_name: Optional[str] = None
        self.snapshots: List[ReportSnapshot] = []
        self.html_path: Optional[str] = None

    # ---- 設定系 -----------------------------------------------------------

    def set_video_name(self, name: str) -> None:
        """HTML上部に表示するファイル名（動画ファイル名など）"""
        self.video_name = name

    # ---- ビルド系 ---------------------------------------------------------

    def build_from_snapshots(
        self,
        snapshots: Sequence[ReportSnapshot],
        basename: str = "auto_report.html",
    ) -> str:
        """
        ReportSnapshot のリストから HTML を生成し、ファイルパスを返す。
        """
        self.snapshots = list(snapshots)
        return self._build(basename)

    # ---- 内部処理 ---------------------------------------------------------

    def _build(self, basename: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        html_path = os.path.join(self.output_dir, basename)
        self.html_path = html_path

        title = self.video_name or "マルチモーダル識別レポート"

        lines: List[str] = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='ja'>")
        lines.append("<head>")
        lines.append("  <meta charset='utf-8'>")
        lines.append(f"  <title>{html.escape(title)}</title>")
        # 簡易CSS
        lines.append("  <style>")
        lines.append("    body { font-family: sans-serif; padding: 16px; }")
        lines.append("    h1 { font-size: 20px; margin-bottom: 8px; }")
        lines.append("    h2 { font-size: 18px; margin-top: 24px; margin-bottom: 8px; }")
        lines.append("    .filename { font-weight: bold; }")
        lines.append("    .snapshot-row { display: flex; gap: 12px; margin-bottom: 8px; }")
        lines.append("    .card { border: 1px solid #ccc; padding: 8px; border-radius: 4px; }")
        # すべて同じ表示サイズにそろえる（ここでは 260x260 の正方形）
        lines.append("    .card img { "
                     "width: 260px; "
                     "height: 260px; "
                     "object-fit: contain; "  # 全体を収める（黒帯的な余白あり）
                     "background: #000; "
                     "display: block; "
                     "margin-bottom: 4px; "
                     "}")
        lines.append("    .rank-label { font-weight: bold; margin-bottom: 4px; }")
        lines.append("    .sims { font-size: 12px; line-height: 1.4; }")
        lines.append("    hr { margin: 16px 0; border: none; border-top: 1px solid #ddd; }")
        lines.append("  </style>")
        lines.append("</head>")
        lines.append("<body>")

        lines.append(f"  <h1>{html.escape(title)}</h1>")
        if self.video_name:
            lines.append(
                f"  <p>ファイル：<span class='filename'>{html.escape(self.video_name)}</span></p>"
            )

        # 各スナップショット
        for snap in self.snapshots:
            lines.append(f"  <h2>{html.escape(snap.time_label)}</h2>")
            lines.append("  <div class='snapshot-row'>")
            for item in snap.items:
                # 実ファイルパス
                img_full = os.path.join(self.output_dir, item.image_path)

                # 画像を base64 にエンコード
                data_uri = ""
                try:
                    with open(img_full, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    # JPEG前提。PNGなら image/png に変える
                    data_uri = f"data:image/jpeg;base64,{b64}"
                except Exception:
                    data_uri = ""

                lines.append("    <div class='card'>")
                lines.append(f"      <div class='rank-label'>{item.rank} </div>") # 位

                if data_uri:
                    lines.append(f"      <img src='{data_uri}' alt='rank {item.rank}'>")
                else:
                    lines.append("      <div class='noimage'>画像なし</div>")

                if self.show_sim:
                    lines.append("      <div class='sims'>")
                    lines.append(f"        <div>Face_sim: {self._fmt_sim(item.face_sim)}</div>")
                    lines.append(f"        <div>App_sim:  {self._fmt_sim(item.app_sim)}</div>")
                    lines.append(f"        <div>Gait_sim: {self._fmt_sim(item.gait_sim)}</div>")
                    lines.append("      </div>")

                lines.append("    </div>")
            lines.append("  </div>")
            lines.append("  <hr>")

        lines.append("</body>")
        lines.append("</html>")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
        try:
            for snap in self.snapshots:
                for item in snap.items:
                    img_full = os.path.join(self.output_dir, item.image_path)
                    if os.path.exists(img_full):
                        os.remove(img_full)
        except Exception:
            # 削除に失敗してもレポート自体は有効なので握りつぶす
            pass

        return html_path

    @staticmethod
    def _fmt_sim(v: Optional[float]) -> str:
        """None/NaN のときは '------'、それ以外は小数第5位まで"""
        if v is None:
            return "------"
        try:
            f = float(v)
        except Exception:
            return "------"
        if math.isnan(f):
            return "------"
        return f"{f:.5f}"
