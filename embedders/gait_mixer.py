from functools import partial
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
import numpy as np
import logging
from ..config import GAIT_MIXER_MODEL_PATH, TTA_FLIP_ENABLED, TTA_NOISE_ENABLED, TTA_TEMPORAL_ENABLED, COCO_LR_PAIRS
from ..utils.gait_mixer_utils import GaitUtils

log = logging.getLogger("app.gait_mixer")

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.training:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_frame=31):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=.01)
        self.ref_pad = torch.nn.ReflectionPad2d((0, 0, kernel_frame-1, 0))
        # self.conv = torch.nn.Conv2d(in_channels=in_channels,
        #                         out_channels=out_channels,
        #                         kernel_size=(31, 1),
        #                         stride=1,
        #                         padding=0)
        self.conv = DepthwiseSeparableConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=(kernel_frame, 1),
                                           padding=0)
        self.act = torch.nn.GELU()
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.no_diff_c = False
        if in_channels != out_channels:
            self.linear = torch.nn.Linear(in_channels, out_channels)
            self.linear_act = torch.nn.GELU()
            self.bn_skip = torch.nn.BatchNorm2d(out_channels, eps=0.001)
            self.no_diff_c = True

    def forward(self, x):
        if self.no_diff_c:
            res_x = rearrange(x, 'b e f j -> b f j e')
            res_x = self.linear(res_x)
            res_x = rearrange(res_x, 'b f j e -> b e f j')
            res_x = self.linear_act(res_x)
            res_x = self.bn_skip(res_x)
        else:
            res_x = x

        x = self.dropout(x)
        x = self.ref_pad(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x += res_x
        return x


class SpatialTransformerTemporalConv(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=3, spatial_embed_dim=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None, out_dim=124, kernel_frame=31):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.spatial_embed_dim = spatial_embed_dim
        self.final_embed_dim = 256 #spatial_embed_dim #8*spatial_embed_dim # spatial_embed_dim * num_joints
        self.out_dim = out_dim

        # spatial patch embedding
        self.spatial_joint_to_embedding = nn.Linear(
            in_chans, spatial_embed_dim)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, spatial_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.spatial_norm = norm_layer(spatial_embed_dim)

        self.conv1 = ConvBlock(
            in_channels=32, out_channels=32, kernel_frame=kernel_frame)
        self.conv2 = ConvBlock(
            in_channels=32, out_channels=64, kernel_frame=kernel_frame)
        self.conv3 = ConvBlock(
            in_channels=64, out_channels=128, kernel_frame=kernel_frame)
        self.conv4 = ConvBlock(
            in_channels=128, out_channels=self.final_embed_dim, kernel_frame=kernel_frame)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=(num_frame, num_joints), stride=1)
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.final_embed_dim),
            nn.Linear(self.final_embed_dim, self.out_dim),
        )

    def spatial_transformer(self, x):
        for blk in self.spatial_blocks:
            x = blk(x)
        return self.spatial_norm(x)

    def spatial_forward(self, x):
        b, f, j, d = x.shape
        x = rearrange(x, 'b f j d -> (b f) j  d')
        x = self.spatial_joint_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) j e -> b e f j', f=f)
        return x

    def temporal_forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        return x

    def forward(self, x):
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        x = self.head(x)
        # x = F.normalize(x, dim=1, p=2)
        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=3, spatial_embed_dim=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None, out_dim=124):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.spatial_embed_dim = spatial_embed_dim
        self.temporal_embed_dim = spatial_embed_dim #* num_joints
        self.out_dim = out_dim 

        # spatial patch embedding
        self.spatial_joint_to_embedding = nn.Linear(
            in_chans, spatial_embed_dim)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, spatial_embed_dim))

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frame+1, self.temporal_embed_dim))
        self.temporal_cls_token = nn.Parameter(
            torch.randn(1, 1, self.temporal_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.temporal_blocks = nn.ModuleList([
            Block(
                dim=self.temporal_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.spatial_norm = norm_layer(spatial_embed_dim)
        self.temporal_norm = norm_layer(self.temporal_embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(self.temporal_embed_dim* num_joints),
            nn.Linear(self.temporal_embed_dim* num_joints, self.out_dim),
        )

    def spatial_transformer(self, x):
        for blk in self.spatial_blocks:
            x = blk(x)
        return self.spatial_norm(x)

    def temporal_transformer(self, x):
        for blk in self.temporal_blocks:
            x = blk(x)
        return self.temporal_norm(x)

    def spatial_forward(self, x):
        b, f, j, d = x.shape
        x = rearrange(x, 'b f j d -> (b f) j  d')
        x = self.spatial_joint_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) j e -> (b j) f e', f=f)
        return x

    def temporal_forward(self, x):
        b, f, e = x.shape
        cls_tokens = repeat(self.temporal_cls_token, '() 1 e -> b 1 e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)

        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b j) f e -> b f (j e)', j=17)
        x = x[:, 0, :]
        return x

    def forward(self, x):
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        x = self.head(x)
        # x = F.normalize(x, dim=1, p=2)
        return x
    
class GaitMixerRunner:
    def __init__(self):
        self.model = SpatialTransformerTemporalConv(
            num_frame=32, num_joints=17, in_chans=2,
            spatial_embed_dim=32, depth=4, num_heads=8, out_dim=128
        )
        state = torch.load(GAIT_MIXER_MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        
        self.gutils = GaitUtils() # ゲートユーティリティ

        # 学習（特徴抽出）用の 2D キーポイントの時系列バッファ
        # 形状の積み上げイメージ：[(17,2), (17,2), ...] 長さ最大32
        self.seq_buffer: list[np.ndarray] = []
        # 直近の埋め込み（32フレーム揃って forward できたときに更新）
        self.prev_feat: np.ndarray | None = None
        # 識別用ギャラリー（1人ぶん前提：形状 (D,)）
        self.gallery: np.ndarray | None = None

        log.info(f"[GAIT] Loaded model from {GAIT_MIXER_MODEL_PATH}")

    def consume_kpts(self, kpts: np.ndarray):
        """
        YOLO-Poseのキーポイントから「対象1人」の1フレームぶんを取り込み、
        32フレーム揃ったら埋め込みを更新して返す。
        Args:
            kpts: (1, K, 3) or (N, K, 3) を想定。先頭の人物のみ使用。
        Returns:
            emb (np.ndarray shape=(D,)) or None
        """
        if kpts is None or kpts.shape[0] == 0:
            return None

        # 先頭人物のみ取り出し（品質ゲート：下半身/半身欠損フレームは採用しない）
        k_one = np.asarray(kpts[0], dtype=np.float32)  # (K,3) or (K,2)
        try:
            # pose_passes_gate は (K,3) でも (K,2) でも動作（confが無い場合は conf=1 扱い）
            if not self.gutils.pose_passes_gate(k_one):
                log.info("[GAIT] frame skipped by pose gate (lower/side missing)")
                return None
        except Exception as ex:
            # ゲート関連で例外が出た場合は安全側でスキップ
            log.info(f"[GAIT] pose gate error -> frame skipped: {ex}")
            return None

        # x,y を (17,2) に整形（17未満は0埋め、超過は先頭17）
        pts = k_one[:, :2].astype(np.float32)  # (K,2)
        if pts.shape[0] >= 17:
            pts = pts[:17]
        else:
            pad = np.zeros((17 - pts.shape[0], 2), np.float32)
            pts = np.concatenate([pts, pad], axis=0)

        # 時系列バッファへ
        self.seq_buffer.append(pts)
        if len(self.seq_buffer) > 32:
            self.seq_buffer.pop(0)

        # 32フレームに到達したら埋め込みを更新
        if len(self.seq_buffer) == 32:
            arr = np.stack(self.seq_buffer, axis=0).astype(np.float32)  # (32,17,2)

            # --- ここが切替ポイント（config 連動） ---
            # 既存の挙動は維持しつつ、時間反転TTAがある場合にも True になるように保険

            use_tta = bool(TTA_FLIP_ENABLED or TTA_NOISE_ENABLED or TTA_TEMPORAL_ENABLED)

            if use_tta:
                # 正規化+TTA（内部で normalize → noise TTA → flip/temporal TTA → embed）
                emb = self.embed_xy_with_tta(arr)
            else:
                # TTAなし：正規化 → モデル素通し
                arr_norm = self.gutils.normalize_pose(arr)
                emb = self._embed_xy_sequence(arr_norm)

            if emb is None:
                log.info("[GAIT] feat update skipped (embed None)")
                return None

            self.prev_feat = emb.astype(np.float32, copy=False)
            log.info(f"[GAIT] feat updated (T=32, dim={emb.shape[0]}, TTA={use_tta})")
            return self.prev_feat

        else:
            log.info(f"[GAIT] buffering {len(self.seq_buffer)}/32 frames")
            return None

    
    def clear_buffer(self) -> int:
        """
        特徴抽出用のフレームバッファを破棄する。
        Returns:
            破棄したフレーム数（元の長さ）
        """
        n = len(self.seq_buffer)
        self.seq_buffer.clear()
        self.prev_feat = None
        log.info(f"[GAIT] feature buffer cleared (removed {n} frames)")
        return n
    
    def build_gallery_from_buffer(self, normalize: bool = False) -> int:
        """
        現在の学習（特徴抽出）状態から識別用ギャラリーを1件構築する。
        優先順位:
        1) seq_buffer が十分あれば、最新バッファから TTA込みで再計算して採用
        2) 足りなければ prev_feat を保険として採用

        さらに、代表シーケンス (T,17,2) を self.gallery_seq_xy に保持し、
        可能であれば TTAキャッシュ (units: 128D×本数, concat: 128*本数) を self._gallery_tta_cache に構築する。
        Returns:
            構築できた参照数（0 or 1）
        """
        emb: np.ndarray | None = None
        gallery_seq_xy: np.ndarray | None = None
        self._gallery_tta_cache = None  # 毎回作り直す

        # 1) まずは最新バッファから再計算を試みる（TTA込み）
        if hasattr(self, "seq_buffer") and len(self.seq_buffer) >= 4:
            try:
                # (T,K,2) を生成。17点に成形。
                arr = np.stack(self.seq_buffer, axis=0).astype(np.float32)  # (T,K,2 or 3)
                if arr.ndim != 3:
                    raise ValueError(f"seq_buffer stacked to unexpected shape: {arr.shape}")

                T, K, C = arr.shape
                if C != 2:
                    arr = arr[..., :2]
                    C = 2
                if K < 17:
                    pad = np.zeros((T, 17 - K, 2), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=1)  # -> (T,17,2)
                elif K > 17:
                    arr = arr[:, :17, :]

                # ★ embed_xy_with_tta 内で正規化等を実施している前提（従来通り）
                emb = self.embed_xy_with_tta(arr)  # (D,) or None
                if emb is not None:
                    gallery_seq_xy = arr  # 代表シーケンス保持
                    log.info(f"[GAITMIXER][ident] gallery source: seq_buffer(T={len(self.seq_buffer)})")
            except Exception as ex:
                log.error(f"[GAITMIXER][ident] gallery build from buffer failed: {ex}")

        # 2) バッファ再計算ができなかった/短い場合は prev_feat を保険で使う
        if emb is None and getattr(self, "prev_feat", None) is not None:
            try:
                emb = self.prev_feat.astype(np.float32, copy=False)
                log.info("[GAITMIXER][ident] gallery source: prev_feat")
            except Exception as ex:
                log.error(f"[GAITMIXER][ident] prev_feat fallback failed: {ex}")
                emb = None

        # 3) どちらも無理なら終了
        if emb is None:
            log.info("[GAITMIXER][ident] gallery build skipped: insufficient frames")
            self.gallery = None
            self.gallery_seq_xy = None
            self._gallery_tta_cache = None
            return 0

        # if normalize:
        #     emb = emb / (np.linalg.norm(emb) + 1e-6)

        # 4) ギャラリー埋め込みを確定
        self.gallery = emb.astype(np.float32, copy=False)  # (D,)
        self.gallery_seq_xy = gallery_seq_xy  # (T,17,2) or None（prev_featフォールバックの場合）

        # 5) （任意/推奨）TTAキャッシュを作る：embed_sequence が使える場合のみ
        #    - units: 各TTA(例 "normal","flip","rev","rev_flip")で 128D を作成したリスト
        #    - concat: units を結合（128*本数）し、結合後に1回だけL2正規化
        try:
            if self.gallery_seq_xy is not None:
                # config から TTA_MODES を取得（なければデフォルト）
                try:
                    from config import TTA_MODES
                except Exception:
                    TTA_MODES = ["normal", "flip", "rev"]

                # embed_sequence の取得（selfに無ければ self.gait から）
                embed_seq_fn = getattr(self, "embed_sequence", None)
                if embed_seq_fn is None and hasattr(self, "gait"):
                    embed_seq_fn = getattr(self.gait, "embed_sequence", None)

                if callable(embed_seq_fn):
                    # 反転・反転+時間反転ユーティリティ（左右ジョイント入替が必要なら既存関数に置換）
                    def _flip_sequence_xy(seq_xy: np.ndarray) -> np.ndarray:
                        a = np.asarray(seq_xy, dtype=np.float32).copy()
                        a[..., 0] = -a[..., 0]  # 正規化座標前提。左右ジョイントの入替が必要ならここで実施。
                        for li, ri in COCO_LR_PAIRS:
                            a[:, [li, ri], :] = a[:, [ri, li], :]
                        return a

                    def _reverse_sequence_xy(seq_xy: np.ndarray) -> np.ndarray:
                        return np.asarray(seq_xy, dtype=np.float32)[::-1].copy()

                    # TTA列生成
                    seqs = []
                    for m in TTA_MODES:
                        if m == "normal":
                            seqs.append(self.gallery_seq_xy)
                        elif m == "flip":
                            seqs.append(_flip_sequence_xy(self.gallery_seq_xy))
                        elif m == "rev":
                            seqs.append(_reverse_sequence_xy(self.gallery_seq_xy))
                        elif m == "rev_flip":
                            seqs.append(_flip_sequence_xy(_reverse_sequence_xy(self.gallery_seq_xy)))
                        else:
                            raise ValueError(f"Unknown TTA mode: {m}")

                    # 128D×本数
                    units: list[np.ndarray] = []
                    for s in seqs:
                        e128 = embed_seq_fn(s)
                        if e128 is None:
                            e128 = np.zeros((128,), dtype=np.float32)
                        units.append(e128.astype(np.float32, copy=False))

                    # 結合後に1回だけ正規化
                    concat = np.concatenate(units, axis=-1).astype(np.float32, copy=False)
                    n = float(np.linalg.norm(concat))
                    if n > 1e-12:
                        concat = concat / n

                    self._gallery_tta_cache = (units, concat)
                    log.info(f"[GAITMIXER][ident] gallery TTA cache built: units={len(units)}, concat_dim={concat.shape[0]}")
                else:
                    # embed_sequence が使えないならキャッシュは見送り
                    self._gallery_tta_cache = None
                    log.info("[GAITMIXER][ident] gallery TTA cache skipped: embed_sequence not available")
            else:
                # prev_feat 由来で代表列が無いケース
                self._gallery_tta_cache = None
                log.info("[GAITMIXER][ident] gallery TTA cache skipped: no gallery_seq_xy")
        except Exception as ex:
            log.warning(f"[GAITMIXER][ident] gallery TTA cache build failed: {ex}")
            self._gallery_tta_cache = None

        log.info(f"[GAITMIXER][ident] gallery built: 1 refs (normalize={normalize})")
        return 1


    def get_gallery(self) -> np.ndarray | None:
        """
        Returns:
            (D,) の1本ベクトル。未構築なら None。
        """
        return self.gallery
    
    def embed_sequence(self, seq: np.ndarray) -> np.ndarray | None:
        """
        任意長 T のキーポイント時系列から埋め込みを生成するユーティリティ。
        受理形状:
            (T, K, 3) or (T, K, 2) or (T, 1, K, 3)
        Returns:
            emb (D,) or None
        """
        if seq is None:
            return None
        a = np.asarray(seq)

        # 形状を (T,K,C) に揃える
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0]  # (T, K, C)
        if a.ndim != 3:
            return None

        T, K, C = a.shape
        if C >= 2:
            xy = a[..., :2].astype(np.float32)  # (T,K,2)
        else:
            return None

        # 関節数を17に合わせる（足りなければ0埋め、超過なら先頭17）
        if K < 17:
            pad = np.zeros((T, 17 - K, 2), np.float32)
            xy = np.concatenate([xy, pad], axis=1)
        elif K > 17:
            xy = xy[:, :17, :]
            
        xy = self.gutils.normalize_pose(xy)

        return self._embed_xy_sequence(xy)  # (D,) or None


    # ---- 内部ヘルパ ----

    def _embed_xy_sequence(self, xy: np.ndarray) -> np.ndarray | None:
        """
        (T,17,2) を 32 フレームへ時間補間し、モデルで埋め込みを計算。
        Returns: (D,) or None
        """
        if xy is None or xy.ndim != 3 or xy.shape[1:] != (17, 2):
            return None
        T = xy.shape[0]
        if T <= 1:
            return None

        # 32フレームへ線形補間
        def _resample_32(x: np.ndarray) -> np.ndarray:
            # x: (T,17,2)
            t_src = np.arange(T, dtype=np.float32)
            t_dst = np.linspace(0, T - 1, 32, dtype=np.float32)
            out = np.empty((32, 17, 2), np.float32)
            for j in range(17):
                out[:, j, 0] = np.interp(t_dst, t_src, x[:, j, 0])
                out[:, j, 1] = np.interp(t_dst, t_src, x[:, j, 1])
            return out

        seq32 = _resample_32(xy) if T != 32 else xy.astype(np.float32, copy=False)
        x = torch.from_numpy(seq32[None])  # (1,32,17,2)
        with torch.no_grad():
            feat = self.model(x)           # (1,D)
        emb = feat.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        # L2 正規化
        # emb = emb / (np.linalg.norm(emb) + 1e-6)
        return emb
    
    def embed_xy_with_tta(self, xy: np.ndarray) -> np.ndarray | None:
        """
        入力: (T,17,2) の未正規化座標
        前処理: normalize_pose
        TTA  : noise(任意) を内包しつつ、flip / temporal を組み合わせて平均
            - flipのみON        → [原, flip] の2系統平均
            - temporalのみON    → [原, 逆順] の2系統平均
            - 両方ON            → [原, flip, 逆順, 逆順+flip] の最大4系統平均
            - どれもOFF         → 単発（原系列のみ）
        出力: (D,)
        """
        if xy is None or xy.ndim != 3 or xy.shape[1:] != (17, 2):
            return None

        # 1) 正規化（方法B）
        seq_norm = self.gutils.normalize_pose(xy)  # (T,17,2)

        # 2) 単発埋め込み（32リサンプル＋モデル）をラップ
        def _embed_once(a_xy: np.ndarray) -> np.ndarray | None:
            return self.embed_sequence(a_xy)  # (T,17,2) 受け

        # 3) “ノイズTTAあり/なし”を内包する埋め込み関数
        def _embed_with_optional_noise(a_xy: np.ndarray) -> np.ndarray | None:
            if TTA_NOISE_ENABLED:
                return self.gutils.noise_tta_embed(a_xy, _embed_once)
            out = _embed_once(a_xy)
            return np.asarray(out, dtype=np.float32).reshape(-1) if out is not None else None

        # 4) TTAバリアント生成
        variants: list[np.ndarray] = []

        # 原系列
        variants.append(seq_norm)

        # 左右反転（座標xの符号反転＋左右ペアの入替）
        if TTA_FLIP_ENABLED:
            flipped = seq_norm.copy()
            flipped[..., 0] = -flipped[..., 0]
            for li, ri in COCO_LR_PAIRS:
                flipped[:, [li, ri], :] = flipped[:, [ri, li], :]
            variants.append(flipped)

        # 時間反転（逆順）
        if TTA_TEMPORAL_ENABLED:
            rev = seq_norm[::-1].copy()
            variants.append(rev)

            # 逆順＋左右反転（両方ON時のみ）
            if TTA_FLIP_ENABLED:
                rev_flip = rev.copy()
                rev_flip[..., 0] = -rev_flip[..., 0]
                for li, ri in COCO_LR_PAIRS:
                    rev_flip[:, [li, ri], :] = rev_flip[:, [ri, li], :]
                variants.append(rev_flip)
            variants.append(rev)

            # 逆順＋左右反転（両方ON時のみ）
            if TTA_FLIP_ENABLED:
                rev_flip = rev.copy()
                rev_flip[..., 0] = -rev_flip[..., 0]
                for li, ri in COCO_LR_PAIRS:
                    rev_flip[:, [li, ri], :] = rev_flip[:, [ri, li], :]
                variants.append(rev_flip)

        # 5) 各バリアントを埋め込んで平均
        feats = []
        for v in variants:
            f = _embed_with_optional_noise(v)
            if f is not None:
                feats.append(f)

        if not feats:
            return None

        feat = np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32, copy=False)
        
        n = float(np.linalg.norm(feat)) + 1e-12 
        if n > 0:
            feat = feat / n
            
        return feat

    def embed_xy_query_variants(self, xy: np.ndarray) -> list[np.ndarray]:
        """
        クエリ用: (T,17,2) 未正規化座標から、設定に応じてクエリ埋め込みを複数本返す。
        - TTA_FLIP_ENABLED=OFF, TTA_TEMPORAL_ENABLED=OFF: [原]（1本）
        - TTA_FLIP_ENABLED=ON , TTA_TEMPORAL_ENABLED=OFF: [原, flip]（2本）
        - TTA_FLIP_ENABLED=OFF, TTA_TEMPORAL_ENABLED=ON : [原, rev]（2本）
        - TTA_FLIP_ENABLED=ON , TTA_TEMPORAL_ENABLED=ON : [原, flip, rev, rev+flip]（4本）
        各バリアント内で TTA_NOISE_ENABLED=True の場合はノイズTTA平均で1本に安定化。
        戻り値: List[(D,)]  ※Noneは除外
        """
        if xy is None or xy.ndim != 3 or xy.shape[1:] != (17, 2):
            return []

        # 設定取得（ローカルimportで後方互換）
        try:
            from config import COCO_LR_PAIRS, TTA_FLIP_ENABLED, TTA_NOISE_ENABLED, TTA_TEMPORAL_ENABLED
        except Exception:
            COCO_LR_PAIRS = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
            try:
                from config import TTA_FLIP_ENABLED
            except Exception:
                TTA_FLIP_ENABLED = False
            try:
                from config import TTA_NOISE_ENABLED
            except Exception:
                TTA_NOISE_ENABLED = False
            try:
                from config import TTA_TEMPORAL_ENABLED
            except Exception:
                TTA_TEMPORAL_ENABLED = False

        # 1) 正規化（方法B）
        seq_norm = self.gutils.normalize_pose(xy)  # (T,17,2)

        # 2) 単発埋め込み
        def _embed_once(a_xy: np.ndarray) -> np.ndarray | None:
            return self.embed_sequence(a_xy)  # (T,17,2) 受け

        # 3) ノイズTTA（有効時のみ平均）
        def _embed_with_optional_noise(a_xy: np.ndarray) -> np.ndarray | None:
            if TTA_NOISE_ENABLED:
                return self.gutils.noise_tta_embed(a_xy, _embed_once)
            out = _embed_once(a_xy)
            return np.asarray(out, dtype=np.float32).reshape(-1) if out is not None else None

        # 4) バリアント生成（設定に忠実）
        variants: list[np.ndarray] = []

        # 原
        variants.append(seq_norm)

        # flip
        if TTA_FLIP_ENABLED:
            flipped = seq_norm.copy()
            flipped[..., 0] = -flipped[..., 0]
            for li, ri in COCO_LR_PAIRS:
                flipped[:, [li, ri], :] = flipped[:, [ri, li], :]
            variants.append(flipped)

        # rev
        if TTA_TEMPORAL_ENABLED:
            rev = seq_norm[::-1]
            variants.append(rev)

            # rev+flip（両方ONのときのみ）
            if TTA_FLIP_ENABLED:
                rev_flip = rev.copy()
                rev_flip[..., 0] = -rev_flip[..., 0]
                for li, ri in COCO_LR_PAIRS:
                    rev_flip[:, [li, ri], :] = rev_flip[:, [ri, li], :]
                variants.append(rev_flip)

        # 5) 各バリアントを埋め込み
        feats: list[np.ndarray] = []
        for v in variants:
            f = _embed_with_optional_noise(v)
            if f is not None:
                feats.append(f.astype(np.float32, copy=False))

        return feats

    def get_learning_progress(self) -> tuple[int, int]:
        """
        特徴抽出バッファの進捗を返す。
        Returns:
            (cur, total) … 現在フレーム数, 目標フレーム数(固定:32)
        """
        cur = int(len(self.seq_buffer)) if hasattr(self, "seq_buffer") else 0
        total = 32  # モデルの学習窓
        return cur, total