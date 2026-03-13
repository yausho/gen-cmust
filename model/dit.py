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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
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

    def forward(self, x, c, T_pred=1):
        shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        x_s, spatial_attn_map = self.spatial_attn(modulate(self.norm1_s(x), shift_s, scale_s))
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

class SimpleConditionEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, history_len=12, patch_size=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * history_len, hidden_size, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, condition):
        if condition.dim() == 4:  # (B, C, H, W) single-frame: repeat to fill conv's expected T
            expected_T = self.conv[0].in_channels // condition.shape[1]
            condition = condition.unsqueeze(1).repeat(1, expected_T, 1, 1, 1)
        B, T, C, H, W = condition.shape
        x = condition.view(B, T * C, H, W) 
        x = self.conv(x)
        return x

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
        self.condition_encoder = SimpleConditionEncoder(in_channels, hidden_size, history_len=history_len, patch_size=patch_size)
        
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
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
        # Fall back to square assumption only when caller provides no shape info.
        if H_patch is None or W_patch is None:
            h = w = int(round(x.shape[1] ** 0.5))
        else:
            h, w = H_patch, W_patch
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, x, t, condition, **kwargs):
        is_seq = False
        T_pred = 1

        if x.dim() == 5:
            is_seq = True
            B, T_pred, C, H, W = x.shape
            x_flat = x.reshape(B * T_pred, C, H, W)
        else:
            B, C, H, W = x.shape
            x_flat = x

        cond_features = self.condition_encoder(condition) 
        cond_spatial = cond_features.flatten(2).transpose(1, 2)

        x_emb = self.x_embedder(x_flat).flatten(2).transpose(1, 2)
        
        # Derive true patch grid from actual input spatial dims (supports non-square H×W).
        actual_patches = x_emb.shape[1]
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        if actual_patches != self.pos_embed.shape[1]:
            # Spatial size differs from input_size — must interpolate.
            # NEVER cache the interpolated result: the cached tensor's autograd graph
            # is freed after backward(), so subsequent forward passes would treat it
            # as a constant leaf, cutting all gradient flow to self.pos_embed.
            H_train = self._train_H_patch
            W_train = self._train_W_patch
            pos_embed_2d = self.pos_embed.transpose(1, 2).view(1, self.hidden_size, H_train, W_train)
            pos_embed_interp = F.interpolate(pos_embed_2d, size=(H_patch, W_patch), mode='bicubic', align_corners=False)
            dynamic_pos_embed = pos_embed_interp.view(1, self.hidden_size, -1).transpose(1, 2)
        else:
            # No interpolation needed — direct reference to the Parameter keeps
            # gradients flowing and values in sync with optimizer updates.
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

        total_attn_scores = 0
        num_blocks = len(self.blocks)
        for idx, block in enumerate(self.blocks):
            x_emb, attn_map = block(x_emb, c, T_pred=T_pred if is_seq else 1)
            score = attn_map.mean(dim=1).sum(dim=1)
            if idx < num_blocks - 2:
                score = score.detach()
            total_attn_scores = total_attn_scores + score

        importance_scores = F.softmax((total_attn_scores / num_blocks) / 2.0, dim=-1)
        x_out = self.unpatchify(self.final_layer(x_emb, c), H_patch=H_patch, W_patch=W_patch)

        if is_seq:
            x_out = x_out.view(B, T_pred, C, H, W)
            importance_scores = importance_scores.view(B, T_pred, -1)
            
        return x_out, importance_scores