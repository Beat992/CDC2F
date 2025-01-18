import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt
import torchjpeg.dct as dct
from model.help_function import inverse_zigzag


class SpaFreInteractionFusionModule(nn.Module):
    """
        func: Interacting and fusing frequency  and spatial features
        input: spatial features: bs * 64 * 128 * 128
               and
               frequency features: bs_16, patch_n(64), 192
    """

    def __init__(self, patch_size):
        super(SpaFreInteractionFusionModule, self).__init__()
        self.t1 = None
        self.t0 = None
        self.patch_size = patch_size
        self.spa_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
        )
        self.fre_conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),
        )

        self.conv_attn_self = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
        )

        self.conv_attn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
        )
        self.conv_prob_attn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
        )
        self.conv_add = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
        )

        self.addchannel_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

    def _feature_enhance(self, x_ori, x_extra, prob_loacl):
        attn_extra = self.conv_attn(x_extra)
        # attn_extra = torch.sigmoid(self.conv_prob_attn(torch.cat([prob_loacl, attn_extra], dim=1)))
        attn_extra = torch.sigmoid(self.conv_prob_attn(prob_loacl + attn_extra))
        attn_self = torch.sigmoid(self.conv_attn_self(x_ori))
        x_ori1 = x_ori * (0.7 * attn_self + 0.3 * attn_extra) + x_ori
        add_extra = self.conv_add(x_extra)
        x_ori2 = x_ori1 + add_extra
        return x_ori2

    def _single_time_forward_(self, X_spa, X_fre, coarse_mask):
        self.spa_feature = X_spa
        self.fre_feature = X_fre
        # X_spa = self.min_max_normalization(X_spa)
        X_spa = self.spa_conv(X_spa)  # .view(b, 8, -1).contiguous()
        X_fre = self.fre_conv(X_fre)  # .view(b, 8, -1).contiguous()

        X_spa = self._feature_enhance(X_spa, X_fre, coarse_mask)
        X_fre = self._feature_enhance(X_fre, X_spa, coarse_mask)
        return torch.cat([X_spa, X_fre], dim=1)

    @staticmethod
    def min_max_normalization(tensor):
        # 获取最后一维的最小值和最大值
        min_vals = tensor.min(dim=(-1, -2), keepdim=True)[0]
        max_vals = tensor.max(dim=(-1, -2), keepdim=True)[0]

        # 进行最小-最大归一化
        normalized_tensor = 2 * (tensor - min_vals) / (max_vals - min_vals) - 1

        return normalized_tensor

    # def forward(self, x_spa, idx):
    def forward(self, x_spa, x_fre, coarse_mask, idx):
        # idx = idx.view(-1, 1, 1, 1)
        # x_spa = F.interpolate(x_spa, 256, mode='bilinear', align_corners=True)
        b, c, h, w = x_spa.shape
        x_spa = x_spa.view(-1, 1, h, w)
        x_spa = F.unfold(x_spa, kernel_size=(self.patch_size//2, self.patch_size//2), padding=0,
                         stride=(self.patch_size//2, self.patch_size//2))
        x_spa = x_spa.view(b, c, self.patch_size//2, self.patch_size//2, -1).permute(0, 4, 1, 2, 3). \
            reshape(-1, c, self.patch_size//2, self.patch_size//2)
        t0_spa, t1_spa = torch.chunk(x_spa, chunks=2, dim=1)
        t0_spa, t1_spa = torch.index_select(t0_spa, 0, idx), torch.index_select(t1_spa, 0, idx)
        t0_fre, t1_fre = torch.chunk(x_fre, chunks=2, dim=1)
        coarse_mask = coarse_mask.view(idx.numel(), 1, self.patch_size, self.patch_size)
        coarse_mask = F.interpolate(coarse_mask, self.patch_size//2)
        t0 = self._single_time_forward_(t0_spa, t0_fre, coarse_mask)
        t1 = self._single_time_forward_(t1_spa, t1_fre, coarse_mask)
        self.t0 = t0
        self.t1 = t1
        t0 = self.addchannel_conv(t0)
        t1 = self.addchannel_conv(t1)

        return t0, t1


if __name__ == '__main__':
    spa = torch.randn([1, 128, 128, 128])
    fre = torch.randn([64, 12, 32, 32])
    prob = torch.randn([64, 1, 16, 16])
    fine_idx = torch.arange(0, 64)
    net = SpaFreInteractionFusionModule(32)
    t0, t1 = net(spa, fre, prob, fine_idx)
