import torch
import torch.nn as nn


class TwoLayerGroupConv(nn.Module):
    def __init__(self, patch_size, dct_size, if_cat=False):
        super(TwoLayerGroupConv, self).__init__()
        if if_cat:
            factor = 2
        else:
            factor = 1
        self.conv = nn.Sequential(
            nn.Conv2d(6 * dct_size ** 2 * factor, 6 * dct_size ** 2 * 8, 3, 1, 1, groups=3 * dct_size ** 2),
            nn.BatchNorm2d(6 * dct_size ** 2 * 8),
            nn.ReLU(),
            nn.Conv2d(6 * dct_size ** 2 * 8, 6 * dct_size ** 2, 3, 1, 1, groups=3 * dct_size ** 2)
        )

    def forward(self, x):
        return self.conv(x)


class FIM(nn.Module):
    def __init__(self,
                 dct_size=4,
                 patch_size=64):
        super(FIM, self).__init__()

        self.fie_conv_t0 = TwoLayerGroupConv(patch_size, dct_size)
        self.fie_conv_t1 = TwoLayerGroupConv(patch_size, dct_size)
        self.fie_conv_attn = TwoLayerGroupConv(patch_size, dct_size)
        self.fie_conv_tov_t0 = TwoLayerGroupConv(patch_size, dct_size)
        self.fie_conv_tov_t1 = TwoLayerGroupConv(patch_size, dct_size)
        self.fie_cat_attn = TwoLayerGroupConv(patch_size, dct_size, True)

    def forward(self, x1, x2):
        x1 = x1.view(-1, self.mini_patch_num, 6 * self.dct_size ** 2).transpose(1, 2).reshape(-1,
                                                                                              6 * self.dct_size ** 2,
                                                                                              self.patch_size // self.dct_size,
                                                                                              self.patch_size // self.dct_size)
        x2 = x2.view(-1, self.mini_patch_num, 6 * self.dct_size ** 2).transpose(1, 2).reshape(-1,
                                                                                              6 * self.dct_size ** 2,
                                                                                              self.patch_size // self.dct_size,
                                                                                              self.patch_size // self.dct_size)
        x1_q = self.conv_t0(x1)
        x2_q = self.conv_t1(x2)
        # attn1 = self.cat_attn(torch.cat([x1_q, x2_q], dim=1))
        attn1 = self.cat_attn(
            torch.cat([x1_q.unsqueeze(2), x2_q.unsqueeze(2)], dim=2).reshape(-1, 12 * self.dct_size ** 2,
                                                                             self.patch_size // self.dct_size,
                                                                             self.patch_size // self.dct_size))
        attn2 = self.conv_attn(x1_q - x2_q)
        attn = torch.sigmoid(attn1 + attn2)
        self.difference_attention_map = attn
        x1 = self.conv_tov_t0(x1) * (attn + 1)
        x2 = self.conv_tov_t1(x2) * (attn + 1)
        x1 = x1.reshape(-1, 6 * self.dct_size ** 2, self.mini_patch_num).transpose(1, 2).reshape(-1,
                                                                                                 self.mini_patch_num,
                                                                                                 6,
                                                                                                 self.dct_size ** 2)
        x2 = x2.reshape(-1, 6 * self.dct_size ** 2, self.mini_patch_num).transpose(1, 2).reshape(-1,
                                                                                                 self.mini_patch_num,
                                                                                                 6,
                                                                                                 self.dct_size ** 2)
        return x1, x2
