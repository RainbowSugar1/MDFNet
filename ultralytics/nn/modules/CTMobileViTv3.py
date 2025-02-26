import torch
import torch.nn as nn

from einops import rearrange
import numpy as np
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn # mg
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)# mg
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MBTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):#depth为2
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MViTBv3(nn.Module):
    def __init__(self, channel, dim, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64*2), dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.mv01 = MV3Block(channel, channel) 
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = MBTransformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)#特征提取
        x = self.conv2(x)#特征提取
        z = x.clone()
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)#后续处理和输出调整
        x = torch.cat((x, z), 1)
        x = self.conv4(x)#后续处理和输出调整
        x = x + y
        x = self.mv01(x)
        return x

class MV3Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup#在 stride == 1 且输入和输出通道数相同的情况下，使用残差连接。

        if expansion == 1:#expansion == 1：只使用深度可分离卷积（depthwise convolution）
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:#expansion > 1：使用标准的 1x1 卷积进行通道扩展后，再使用深度可分离卷积，最后再通过 1x1 卷积将特征映射回输出通道
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.se = SEBlock(oup)
    def forward(self, x):
        if self.use_res_connect:
            return x +self.se(self.conv(x))
        else:
            return self.se(self.conv(x))

# ############################################
# https://github.com/iscyy/ultralyticsPro
import numpy as np
# from ultralytics.nn.modules.block import Conv
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


################# 
class CTMViTBv3(nn.Module):
    def __init__(self, c1, c2, n=1, extra=2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  #self.c用于调整中间通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)#卷积层1
        self.cv2 = Conv(2 * self.c, c2, 1) #卷积层2
        self.m = nn.Sequential(*(MViTBv3(self.c, self.c) for _ in range(n)))
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)#特征图在此处被分割，一部分分为a,一部分分为b，.chunk(2, 1) 将其分为两部分，每部分的通道数为 self.c。最终得到的 a 和 b 的通道数分别是 self.c，通过Self.cv1得到的通道数为2*self.c
        return self.cv2(torch.cat((self.m(a), b), 1))#特征图a被传入mvitbv3,b则直接保留进行concat

