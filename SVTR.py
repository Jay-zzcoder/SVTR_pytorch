import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

import numpy as np
from einops import rearrange, repeat




class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, act=nn.GELU()):
        super(Convblock, self).__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            self.act
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class Merging(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[2, 1], padding=1):
        super(Merging, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = rearrange(out, "n c h w -> n (h w) c")
        return self.norm(out)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, mixer="global", HW=[8, 25], window_size=[7, 11], heads_dim=64, dropout=0.):
        super(Attention, self).__init__()
        self.heads = heads
        self.mixer = mixer
        self.HW = HW
        if HW is not None:
            self.N = self.HW[0] * self.HW[1]
            self.C = dim
        self.project_out = not(heads == 1 and heads_dim == dim)

        self.scale = heads_dim ** -0.5
        self.inner_dims = heads * heads_dim
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, self.inner_dims*3, bias=qkv_bias)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dims, dim),
            nn.Dropout(dropout)) if self.project_out else nn.Identity()
        if HW is not None:
            H = HW[0]
            W = HW[1]
        if mixer == 'Local' and HW is not None:
            hk = window_size[0]
            wk = window_size[1]
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1).cuda()
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                                                               2].flatten(1)
            mask_inf = torch.full([H * W, H * W], float('-inf')).cuda()
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            #print(mask.shape)
            self.mask = mask.unsqueeze(0).unsqueeze(0)
            #print("mask : ", self.mask.shape)
        self.mixer = mixer

    def forward(self, x):
        #print(x.shape)
        if self.HW is not None:
            N, C = self.N, self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.mixer == 'Local':
            dots += self.mask
        atten = self.dropout(self.attend(dots))
        out = torch.matmul(atten, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=1, dropout=0.):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio*dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        out = self.net(x)
        return out


class MixingBlock(nn.Module):
    def __init__(self, dim, heads, heads_dim, mlp_ratio=4, qkv_bias=False, HW=[8, 25], window_size=[7, 11], mixer="global", pre_norm=True):
        super(MixingBlock, self).__init__()
        self.prenorm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.att = Attention(dim, heads, qkv_bias=qkv_bias, HW=HW, window_size=window_size, mixer=mixer, heads_dim=heads_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
    def forward(self, x):
        if self.prenorm:
            x = self.norm1(self.att(x)) + x
            out = self.norm2(self.mlp(x)) + x
        else:
            x = self.att(self.norm1(x)) + x
            out = self.mlp(self.norm2(x)) + x
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channel, embed_dims, norm=None):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size[0] // 4) * (img_size[1] // 4)
        self.conv = nn.Sequential(
            Convblock(in_channels=in_channel, out_channels=embed_dims//2, kernel_size=3, stride=2, padding=1),
            Convblock(in_channels=embed_dims//2, out_channels=embed_dims, kernel_size=3, stride=2, padding=1)
        )

        self.norm = norm

    def forward(self, x):
        out = self.conv(x)
        out = rearrange(out, "n c h w -> n (h w) c")
        if self.norm is not None:
            out = self.norm(out)
        return out


class SVTR_backbone(nn.Module):
    def __init__(self, img_size=[32, 100],
            in_channels=3,
            embed_dim=[64, 128, 256],
            depth=[3, 6, 3],
            num_heads=[2, 4, 8],
            mixer=['Local'] * 6 + ['Global'] *
            6,  # Local atten, Global atten, Conv
            windows_size=[[7, 11], [7, 11], [7, 11]],
            mlp_ratio=4,
            patch_merging=True,
            epsilon=1e-6,
            qkv_bias=True,
            last_drop=0.1,
            out_channels=192,
            out_char_num=32,
            block_unit='MixingBlock',
            act='nn.GELU',
            last_stage=True,
            pre_norm=True,
            use_lenhead=False):
        super(SVTR_backbone, self).__init__()
        self.img_size = img_size
        self.in_channel = in_channels
        self.embed_dim = embed_dim
        self.HW = [img_size[0] // 4, img_size[1] // 4]
        self.patch_emedding = PatchEmbedding(self.img_size, self.in_channel, self.embed_dim[0])
        self.num_patches = self.patch_emedding.num_patches
        self.out_channels = out_channels
        self.patch_merging = patch_merging
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim[0]))
        self.block = eval(block_unit)
        self.prenorm = pre_norm
        #Stage 1
        self.stage1 = nn.ModuleList([
            self.block(embed_dim[0], num_heads[0], embed_dim[0], mixer=mixer[0:depth[0]][i],qkv_bias=qkv_bias, mlp_ratio=mlp_ratio, HW=self.HW, window_size=windows_size[0])
            for i in range(depth[0])
        ])
        if patch_merging:
            self.merging1 = Merging(embed_dim[0], embed_dim[1])
            HW =[self.HW[0] // 2, self.HW[1]]

        #Stage 2
        self.stage2 = nn.ModuleList([
            self.block(embed_dim[1], num_heads[0], embed_dim[1], mixer=mixer[depth[0]:depth[0] + depth[1]][i], qkv_bias=qkv_bias, mlp_ratio=mlp_ratio, HW=HW, window_size=windows_size[1])
            for i in range(depth[1])
        ])
        if patch_merging:
            self.merging2 = Merging(embed_dim[1], embed_dim[2])
            HW =[self.HW[0] // 4, self.HW[1]]

        #Stage 3
        self.stage3 = nn.ModuleList([
            self.block(embed_dim[2], num_heads[0], embed_dim[2], mixer=mixer[depth[0] + depth[1]:][i], qkv_bias=qkv_bias, mlp_ratio=mlp_ratio, HW=HW, window_size=windows_size[2])
            for i in range(depth[2])
        ])

        self.last_stage = last_stage
        if last_stage:
            self.combing = nn.Sequential(
                nn.AdaptiveAvgPool2d([1, out_char_num]),
                nn.Conv2d(embed_dim[2], self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Hardswish(),
                nn.Dropout(last_drop)
            )
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(last_drop)

        trunc_normal_(self.pos_embed, std=.02)
        self.norm = nn.LayerNorm(self.embed_dim[-1])
        #self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)


    def forward_featrures(self, x):
        x = self.patch_emedding(x)
        x = x + self.pos_embed
        #print("patch embedding shape: ", x.shape)
        for blk1 in self.stage1:
            x = blk1(x)
        #print("blk1 shape: ", x.shape)
        if self.patch_merging:
            x = rearrange(x, "b (h w) c -> b c h w", h=self.HW[0])
            x = self.merging1(x)
        for blk2 in self.stage2:
            x = blk2(x)
        #print("stage2 shape: ", x.shape)
        if self.patch_merging:
            x = rearrange(x, "b (h w) c -> b c h w", h=self.HW[0] //2)
            x = self.merging2(x)
        for blk3 in self.stage3:
            x = blk3(x)
        if self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_featrures(x)
        if self.last_stage:
            x = rearrange(x, "n (h w) c-> n c h w", h=self.HW[0] //4)
            x = self.combing(x)
        x = rearrange(x, "n c h w -> n (h w) c")
        return x

class SVTR_head(nn.Module):
    def __init__(self, in_channels=192, char_num=37, mid_channels=None):
        super(SVTR_head, self).__init__()
        self.in_channels = in_channels
        self.out_channels = char_num
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            self.fc = nn.Linear(self.in_channels, self.out_channels)

        else:
            self.fc = nn.Sequential(
                nn.Linear(self.in_channels, self.mid_channels),
                nn.Linear(self.mid_channels, self.out_channels)
            )
    def forward(self, x):
        out = self.fc(x)
        out = rearrange(out, "b t c -> t b c")
        predicts = out.log_softmax(2).requires_grad_()
        return predicts

class SVTR(nn.Module):
    def __init__(self):
        super(SVTR, self).__init__()
        self.backbone = SVTR_backbone()
        self.head = SVTR_head()

    def forward(self, x):
        out = self.head(self.backbone(x))
        #out = rearrange(out, "b t c -> t b c")
        return out







if __name__ == "__main__":
    x = torch.rand(1, 3, 32, 100)
    x1 = torch.rand(1, 3, 10)
    patch_embedding = PatchEmbedding([32,100], 10)
    merging = Merging(3, 10)
    mixingblock = MixingBlock(10, 8, 20, mixer="local")
    svtr = SVTR()
    y = svtr(x)
    print(y.shape)

