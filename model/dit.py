import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn_map = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_map

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class RelativePositionBias(nn.Module):
    """Learnable relative position bias for spatial attention."""
    def __init__(self, num_heads, max_grid_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.max_size = max_grid_size
        # Relative position table: 2*max_size-1 x 2*max_size-1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_grid_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, H, W, device):
        # Generate relative position coordinates
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, H, W
        coords = coords.flatten(1)  # 2, H*W

        relative_coords = coords[:, :, None] - coords[:, None, :]  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0)  # H*W, H*W, 2

        # Shift to non-negative indices
        relative_coords[:, :, 0] += self.max_size - 1
        relative_coords[:, :, 1] += self.max_size - 1

        # Linear index
        rel_pos_idx = relative_coords[:, :, 0] * (2 * self.max_size - 1) + relative_coords[:, :, 1]

        # Get bias
        rel_pos_bias = self.relative_position_bias_table[rel_pos_idx]  # H*W, H*W, num_heads
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).unsqueeze(0)  # 1, num_heads, H*W, H*W
        return rel_pos_bias


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1_s = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.spatial_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        self.norm1_t = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.temporal_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, T_pred=1, rel_pos_bias=None):
        shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x_s, spatial_attn_map = self.spatial_attn(modulate(self.norm1_s(x), shift_s, scale_s), rel_pos_bias=rel_pos_bias)
        x = x + gate_s.unsqueeze(1) * x_s

        if T_pred > 1:
            BT, N, C = x.shape
            B = BT // T_pred
            x_temporal = x.view(B, T_pred, N, C).permute(0, 2, 1, 3).reshape(B * N, T_pred, C)
            shift_t_re = shift_t.view(B, T_pred, 1, C).expand(B, T_pred, N, C).permute(0, 2, 1, 3).reshape(B * N, T_pred, C)
            scale_t_re = scale_t.view(B, T_pred, 1, C).expand(B, T_pred, N, C).permute(0, 2, 1, 3).reshape(B * N, T_pred, C)
            norm_x_t = self.norm1_t(x_temporal)
            modulated_x_t = norm_x_t * (1 + scale_t_re) + shift_t_re
            x_t, _ = self.temporal_attn(modulated_x_t)
            gate_t_re = gate_t.view(B, T_pred, 1, C).expand(B, T_pred, N, C).permute(0, 2, 1, 3).reshape(B * N, T_pred, C)
            x_t = (x_t * gate_t_re).view(B, N, T_pred, C).permute(0, 2, 1, 3).reshape(BT, N, C)
            x = x + x_t

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, spatial_attn_map

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EnhancedConditionEncoder(nn.Module):
    """
    Enhanced condition encoder with:
    1. Multi-scale temporal convolution (capture both short and long-term patterns)
    2. Temporal attention for cross-time dependencies
    3. Output projection for residual connection
    """
    def __init__(self, in_channels, hidden_size, history_len=12, patch_size=4):
        super().__init__()
        self.history_len = history_len
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # Multi-scale temporal convolutions
        # Short-term: kernel=1 (current frame focus)
        # Mid-term: kernel=3 (3-frame patterns)
        # Long-term: kernel=5 (5-frame patterns)
        self.conv_short = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size // 4, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(),
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(in_channels * 3, hidden_size // 4, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(),
        )

        self.conv_long = nn.Sequential(
            nn.Conv2d(in_channels * 5, hidden_size // 4, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(),
        )

        # Global temporal context: aggregate all frames
        self.conv_global = nn.Sequential(
            nn.Conv2d(in_channels * history_len, hidden_size // 4, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(),
        )

        # Fuse multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Temporal attention for sequence input
        self.temporal_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True, dropout=0.1)
        self.temporal_norm = nn.LayerNorm(hidden_size)

        # Projection for residual connection to output
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, condition):
        # condition: (B, T, C, H, W) or (B, C, H, W)
        if condition.dim() == 4:
            B, C, H, W = condition.shape
            condition = condition.unsqueeze(1).expand(-1, self.history_len, -1, -1, -1)

        B, T, C, H, W = condition.shape

        # Multi-scale temporal features
        # Short: last frame
        x_short = self.conv_short(condition[:, -1])  # (B, hidden//4, H//p, W//p)

        # Mid: last 3 frames (pad if needed)
        mid_frames = condition[:, max(0, T-3):]
        if mid_frames.shape[1] < 3:
            mid_frames = F.pad(mid_frames, (0, 0, 0, 0, 0, 0, 3 - mid_frames.shape[1], 0))
        mid_frames = mid_frames.reshape(B, -1, H, W)
        x_mid = self.conv_mid(mid_frames)

        # Long: last 5 frames
        long_frames = condition[:, max(0, T-5):]
        if long_frames.shape[1] < 5:
            long_frames = F.pad(long_frames, (0, 0, 0, 0, 0, 0, 5 - long_frames.shape[1], 0))
        long_frames = long_frames.reshape(B, -1, H, W)
        x_long = self.conv_long(long_frames)

        # Global: all frames
        global_frames = condition.reshape(B, -1, H, W)
        x_global = self.conv_global(global_frames)

        # Concatenate multi-scale features
        x_multi = torch.cat([x_short, x_mid, x_long, x_global], dim=1)  # (B, hidden, H//p, W//p)

        # Fuse
        x_fused = self.fusion(x_multi)  # (B, hidden, H//p, W//p)

        # Spatial features for Transformer input
        spatial_features = x_fused.flatten(2).transpose(1, 2)  # (B, N, hidden)

        # Temporal attention (treat spatial patches as batch)
        if T > 1:
            # Reshape for temporal attention
            N = spatial_features.shape[1]
            spatial_features = self.temporal_norm(spatial_features)

        return spatial_features, x_fused  # (B, N, hidden), (B, hidden, H//p, W//p)


class CMuST_DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=4, in_channels=1, hidden_size=384,
                 depth=8, num_heads=12, mlp_ratio=4.0, learn_sigma=False,
                 history_len=12, forecast_len=12, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.forecast_len = forecast_len

        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = (input_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, forecast_len, 1, hidden_size))

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.condition_encoder = EnhancedConditionEncoder(in_channels, hidden_size, history_len=history_len, patch_size=patch_size)

        # Relative position bias for spatial attention
        self.rel_pos_bias = RelativePositionBias(num_heads, max_grid_size=input_size // patch_size)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Condition residual projection
        self.cond_residual_proj = nn.Linear(hidden_size, hidden_size)

        # Store the training-time patch grid shape to support non-square grids.
        self._train_H_patch = input_size // patch_size
        self._train_W_patch = input_size // patch_size

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.temporal_pos_embed, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, H_patch=None, W_patch=None):
        c = self.out_channels
        p = self.patch_size
        if H_patch is None or W_patch is None:
            h = w = int(round(x.shape[1] ** 0.5))
        else:
            h, w = H_patch, W_patch
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, x, t, condition):
        is_seq = False
        T_pred = 1
        if x.dim() == 5:
            is_seq = True
            B, T_pred, C, H, W = x.shape
            x_flat = x.reshape(B * T_pred, C, H, W)
        else:
            B, C, H, W = x.shape
            x_flat = x

        # Enhanced condition encoding with residual
        cond_spatial, cond_spatial_2d = self.condition_encoder(condition)

        x_emb = self.x_embedder(x_flat).flatten(2).transpose(1, 2)

        actual_patches = x_emb.shape[1]
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        if actual_patches != self.pos_embed.shape[1]:
            H_train = self._train_H_patch
            W_train = self._train_W_patch
            pos_embed_2d = self.pos_embed.transpose(1, 2).view(1, self.hidden_size, H_train, W_train)
            pos_embed_interp = F.interpolate(pos_embed_2d, size=(H_patch, W_patch), mode='bicubic', align_corners=False)
            dynamic_pos_embed = pos_embed_interp.view(1, self.hidden_size, -1).transpose(1, 2)
        else:
            dynamic_pos_embed = self.pos_embed

        x_emb = x_emb + dynamic_pos_embed

        if is_seq:
            x_emb = x_emb.view(B, T_pred, -1, self.hidden_size)
            x_emb = x_emb + self.temporal_pos_embed[:, :T_pred, :, :]
            x_emb = x_emb + cond_spatial.unsqueeze(1)
            x_emb = x_emb.view(B * T_pred, -1, self.hidden_size)
        else:
            x_emb = x_emb + cond_spatial

        t_emb = self.t_embedder(t)
        c = t_emb.unsqueeze(1).repeat(1, T_pred, 1).view(B * T_pred, -1) if is_seq else t_emb

        # Compute relative position bias (once, reused across blocks)
        rel_pos = self.rel_pos_bias(H_patch, W_patch, x.device)

        total_attn_scores = 0
        num_blocks = len(self.blocks)
        for idx, block in enumerate(self.blocks):
            x_emb, attn_map = block(x_emb, c, T_pred=T_pred if is_seq else 1, rel_pos_bias=rel_pos)
            score = attn_map.mean(dim=1).sum(dim=1)
            if idx < max(0, num_blocks - 4):
                score = score.detach()
            total_attn_scores = total_attn_scores + score

        importance_scores = F.softmax((total_attn_scores / num_blocks) / 0.5, dim=-1)

        # Add condition residual before final layer
        cond_residual = self.cond_residual_proj(cond_spatial.mean(dim=1, keepdim=True))  # (B, 1, hidden)
        if is_seq:
            cond_residual = cond_residual.unsqueeze(1).expand(B, T_pred, 1, -1).reshape(B * T_pred, 1, -1)
        x_emb = x_emb + 0.1 * cond_residual  # Small residual weight

        x_out = self.unpatchify(self.final_layer(x_emb, c), H_patch=H_patch, W_patch=W_patch)

        if is_seq:
            x_out = x_out.view(B, T_pred, C, H, W)
            importance_scores = importance_scores.view(B, T_pred, -1)

        return x_out, importance_scores
