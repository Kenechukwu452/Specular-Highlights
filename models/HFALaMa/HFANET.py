import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms as T
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        return self.conv_1(x)

class DFA(nn.Module):
    def __init__(self, dim=36):
        super().__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = 2
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w), mode='nearest')
        y_d = self.lde(y)
        return f + self.linear_2(x_l + y_d)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=3, padding=0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.features(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=3, padding=0, AF=nn.ReLU):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channel),
            AF(),
        )

    def forward(self, x):
        return self.features(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channel, scale_factor):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channel, scale_factor, scale_factor),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)

class PartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_I = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_M = nn.Conv2d(1, 1, kernel_size, padding=padding, dilation=dilation, bias=False)
        nn.init.constant_(self.conv_M.weight, 1.0)
        self.conv_M.requires_grad_(False)

    def forward(self, x, M):
        M = self.conv_M(M)
        index = M == 0
        M[index] = 1
        x = self.conv_I(M * x)
        x = F.relu(self.bn(x / M))
        M = M.masked_fill(index, 0)
        return x, M


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x

def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self, channel=3, k=3):
        super().__init__()
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a))).reshape(b, k, c)
        bar_a = self.softmax(hat_a).unsqueeze(-2)
        return torch.sum(bar_a * x_all, 1).reshape(b, h, w, c)

class S2Attention(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        return self.mlp2(a).permute(0, 3, 1, 2)

class DFAS2(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.DFA = DFA(dim=dim)
        self.s2 = S2Attention(channels=dim)

    def forward(self, x):
        return self.DFA(x) + self.s2(x)
class  HFABlock(nn.Module):

    def __init__(self, channels=3):

        super().__init__()


        self.channels = 3


        self.DFAS2 = DFAS2(dim=3)

        self.conv5 = nn.Conv2d(64, channels, 1)
        self.conv4 = nn.Conv2d(32 + channels, channels * 2, 1)
        self.conv3 = nn.Conv2d(16 + channels * 2, channels * 3, 1)
        self.conv2 = nn.Conv2d(8 + channels * 3, channels * 4, 1)
        self.conv1 = nn.Conv2d(4 + channels * 4, channels * 5, 1)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.conv5(F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True))
        x = torch.cat([x, x4], dim=1)
        x = self.conv4(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        return x


class HFANet(nn.Module):
    """ 高光移除网络 """

    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder1 = EncoderBlock(3, 4, 3, padding=1)
        self.encoder2 = EncoderBlock(4, 8, 3, padding=1)
        self.encoder3 = EncoderBlock(8, 16, 3, padding=1)
        self.encoder4 = EncoderBlock(16, 32, 3, padding=1)
        self.encoder5 = EncoderBlock(32, 64, 3, padding=1)
        # 解码器
        self.decoder5 = DecoderBlock(64, 32, scale_factor=2)
        self.decoder4 = DecoderBlock(64, 16, scale_factor=2)
        self.decoder3 = DecoderBlock(32, 8, scale_factor=2)
        self.decoder2 = DecoderBlock(16, 4, scale_factor=2)
        self.decoder1 = DecoderBlock(8, 1, scale_factor=2)
        # CDFF 模块
        self.HFA =  HFABlock()
        # 输出卷积块
        self.M_conv = nn.Sequential(
            ConvBlock(16, 8, 3, 1),
            ConvBlock(8, 4, 3, 1),
            ConvBlock(4, 1, 3, 1, nn.Sigmoid)
        )
        self.S_conv = nn.Sequential(
            ConvBlock(17, 8, 3, 1),
            ConvBlock(8, 3, 3, 1),
        )
        self.D_conv1 = PartialConvBlock(19, 13, 5, padding=2)
        self.D_conv2 = PartialConvBlock(13, 8, 5, padding=2)
        self.D_conv3 = PartialConvBlock(8, 3, 5, padding=2)

    def forward(self, I):


        x1 = self.encoder1(I)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x_HFA = self.HFA(x1, x2, x3, x4, x5)

        x6 = torch.cat([self.decoder5(x5), x4], dim=1)
        x7 = torch.cat([self.decoder4(x6), x3], dim=1)
        x8 = torch.cat([self.decoder3(x7), x2], dim=1)
        x9 = torch.cat([self.decoder2(x8), x1], dim=1)
        x10 = torch.cat([self.decoder1(x9), x_HFA], dim=1)
        M = self.M_conv(x10)
        S = self.S_conv(torch.cat([x10, M], dim=1))
        D, M_ = self.D_conv1(torch.cat([x10, I - M * S], dim=1), 1 - M)
        D, M_ = self.D_conv2(D, M_)
        D, M_ = self.D_conv3(D, M_)

        return M, S, D
    def predict(self, image: Image.Image, use_gpu=False):
        if image.mode != 'RGB':
            image = image.convert('RGB')


        w, h = image.size
        w_padded = (w//32+(w % 32 != 0))*32
        h_padded = (h//32+(h % 32 != 0))*32
        image_padded = cv.copyMakeBorder(
            np.uint8(image), 0, h_padded-h, 0, w_padded-w, cv.BORDER_REFLECT)
        image = T.ToTensor()(image_padded).unsqueeze(0)
        M, S, D = self(image.to('cpu'))
        M = T.ToPILImage()(M.to('cpu').ge(0.5).to(torch.float32).squeeze())
        S = T.ToPILImage()(S.to('cpu').squeeze())
        D = T.ToPILImage()(D.to('cpu').squeeze())

        M = M.crop((0, 0, w, h))
        S = S.crop((0, 0, w, h))
        D = D.crop((0, 0, w, h))
        return M, S, D

    def remove_specular(self, image: Image.Image):

        return self.predict(image)[-1]


if __name__ == '__main__':
    image = Image.open(r'D:\Specular-Removal-master\a\daxiu_marked\A1.png')
    model = HFANet().to('cuda:0')
    M, S, D = model.predict(image)
    mpl.rc_file('../resource/style/image_process.mplstyle')
    fig, axes = plt.subplots(1, 4, num='高光去除')
    images = [image, M, S, D]
    titles = ['Original image', 'Specular mask',
              'Specular image', 'Specular removal image']
    for ax, im, title in zip(axes, images, titles):
        cmap = plt.cm.gray if title == 'Specular mask' else None
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
    plt.show()





