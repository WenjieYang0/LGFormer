import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math


class DepthAttention(nn.Module):
    def __init__(self, channel):
        super(DepthAttention, self).__init__()

        self.cfc = nn.Parameter(torch.zeros(channel, 2))
        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

    def refine(self, x, eps=1e-5):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(x.var(dim=(2, 3), keepdim=True).clamp(min=eps))
        return torch.cat((mean, std), dim=2)

    def recalibration(self, t):
        t = t.squeeze(-1)
        z = torch.einsum('bci,ci->bc', t, self.cfc).unsqueeze(-1).unsqueeze(-1)
        return self.activation(self.bn(z))

    def forward(self, x):
        w = self.refine(x)
        w = self.recalibration(w)
        return x * w


class TSE(nn.Module):
    def __init__(self, embed_dim, conv_kernel, in_chans, stride=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=(1, conv_kernel), padding=(0, conv_kernel // 2), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
        )
        self.att = DepthAttention(embed_dim)
        self.cproj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(in_chans, 11),
                      padding=(0, 5), stride=(1, stride), bias=False, groups=embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
            Rearrange('b e (h) (w) -> b (h w) e'),

        )

    def forward(self, x):
        x = self.proj(x)
        x = self.att(x)
        x = self.cproj(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model=32, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class LGA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias,
                 attn_drop, proj_drop, local):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_dim = dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
        self.local_perception = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=local, stride=2, padding=local // 2, groups=dim, bias=False),
            nn.BatchNorm1d(dim, eps=1e-5),
        )

    def forward(self, x):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)

        x_ = self.local_perception(x.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.k(x_).reshape(b, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_).reshape(b, -1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class ParallelConv(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_sizes = [5, 9, 25]
        self.proj = nn.ModuleList([
            nn.Conv1d(21, 21, kernel_size=k, padding=k // 2, groups=21, bias=False)
            for k in kernel_sizes
        ])

    def forward(self, x):
        x = torch.chunk(x, len(self.proj), dim=1)
        out = [conv(feat) for conv, feat in zip(self.proj, x)]
        return torch.cat(out, dim=1)


class HybridFFN(nn.Module):
    def __init__(self, in_features, hidden_features, drop):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, 1, 1, 0, bias=False),
            nn.BatchNorm1d(hidden_features, eps=1e-5),
            nn.ELU(),
        )
        self.parallel = ParallelConv()
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_features, eps=1e-5),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Conv1d(hidden_features, in_features, 1, 1, 0, bias=False),
            nn.BatchNorm1d(in_features, eps=1e-5),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.parallel(x)
        x = self.conv2(x)
        return x.permute(0, 2, 1)


class LETBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, qkv_bias, drop, attn_drop,
                 norm_layer, local):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LGA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, local=local)
        self.norm2 = norm_layer(dim)
        self.mlp = HybridFFN(in_features=dim, hidden_features=mlp_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LGFormer(nn.Module):
    def __init__(self, seq_len, num_classes, in_channel, avp_k=65, embed_dims=32, conv_kernel=75,
                 num_heads=4, mlp_dim=63, qkv_bias=True,
                 drop_rate=0.4, attn_drop_rate=0.4, norm_layer=nn.LayerNorm,
                 depths=1, local=9, dp=0.2, stride=2,pool=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims

        self.patch_embed = TSE(in_chans=in_channel, embed_dim=embed_dims, conv_kernel=conv_kernel, stride=stride)
        self.pe = PositionalEncoding(d_model=embed_dims, max_len=int(seq_len / stride))

        self.LET_block = nn.Sequential(*[
            LETBlock(dim=embed_dims, num_heads=num_heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias,
                     drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, local=local)
            for _ in range(depths)
        ])

        self.avp_pad = (avp_k - 1) // 2
        self._avg_pooling = nn.AvgPool1d(kernel_size=avp_k, stride=pool, padding=self.avp_pad)
        self.pooled_len = (seq_len // stride - avp_k + 2 * self.avp_pad) // pool + 1
        self.head = nn.Sequential(
            nn.Dropout(dp),
            nn.Linear(self.pooled_len * embed_dims, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pe(x)
        x = self.LET_block(x)
        x = self._avg_pooling(x.permute(0, 2, 1)).flatten(start_dim=1)
        return self.head(x)
