import torch
import torch.nn.functional as F
import torch.nn as nn
from torchjpeg import dct
from model.help_function import inverse_zigzag, size_restore


class FreFeaturePred(nn.Module):
    def __init__(self,
                 dct_size=4,
                 patch_size=64):
        super(FreFeaturePred, self).__init__()
        self.dct_size = dct_size
        self.patch_size = patch_size

        self.f_conv = nn.Sequential(
            nn.Conv2d(6, 16, 1, 1, 0),
            # nn.Conv2d(3, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
        )

        self.pred = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1, padding_mode='replicate'),
        )

        self.fre_embedding = nn.Sequential(
            nn.Linear(6 * self.dct_size ** 2, 64),
            # nn.Linear(3 * self.dct_size ** 2, 64),
            nn.BatchNorm1d((self.patch_size//self.dct_size)**2),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d((self.patch_size//self.dct_size)**2),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d((self.patch_size//self.dct_size)**2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d((self.patch_size//self.dct_size)**2),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.deconv = nn.Sequential(
            nn.Conv2d(1, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            # nn.Conv2d(64, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1, padding_mode='replicate'),
        )

    def forward(self, t0_fre, t1_fre):
        # <-------IDCT-------->
        t0_fre2spa = self._idct_and_feature_fre_size_restore(t0_fre)  # bs16, 6, 64, 64
        t1_fre2spa = self._idct_and_feature_fre_size_restore(t1_fre)
        x_fre = torch.cat([t0_fre2spa, t1_fre2spa], dim=1)

        # <------空间域预测------->
        t0_fre_spa = self.f_conv(t0_fre2spa)
        t1_fre_spa = self.f_conv(t1_fre2spa)
        pred3_score = torch.abs(t0_fre_spa - t1_fre_spa)
        pred3_score = self.pred(pred3_score)

        # <------频域预测------->
        bs_block = t0_fre.shape[0]
        t0_fre = t0_fre.view(bs_block, -1, 6 * self.dct_size ** 2)
        t1_fre = t1_fre.view(bs_block, -1, 6 * self.dct_size ** 2)
        # t0_fre = t0_fre.view(bs_patch, -1, 3 * self.dct_size ** 2)
        # t1_fre = t1_fre.view(bs_patch, -1, 3 * self.dct_size ** 2)
        t0_fre = self.fre_embedding(t0_fre)
        t1_fre = self.fre_embedding(t1_fre)
        pred3_score2 = self.pred_head(torch.abs(t0_fre - t1_fre))
        pred3_score2 = pred3_score2.view(bs_block, 1, self.patch_size // self.dct_size,
                                         self.patch_size // self.dct_size)

        pred3_score2 = self.deconv(pred3_score2)

        pred3_score = pred3_score + pred3_score2

        return pred3_score, x_fre

    def _idct_and_feature_fre_size_restore(self, x):
        b_patch, n, c, _ = x.shape
        x = x.reshape(b_patch, n * c, -1)
        x = inverse_zigzag(x, dct_size=self.dct_size)
        x = x.view(b_patch, n, c, self.dct_size, self.dct_size)
        x = dct.block_idct(x).reshape(b_patch, n, c, -1)
        x = x.permute(0, 2, 3, 1).reshape(-1, self.dct_size ** 2, n)
        x = F.fold(x, kernel_size=(self.dct_size, self.dct_size), output_size=(self.patch_size, self.patch_size),
                   dilation=1, padding=0, stride=(self.dct_size, self.dct_size))
        x = x.view(b_patch, c, -1)
        min_value = torch.min(x, dim=2, keepdim=True)[0]
        max_value = torch.max(x, dim=2, keepdim=True)[0]
        # 将张量归一化到[-1, 1]的范围
        x = 2 * (x - min_value) / (max_value - min_value) - 1
        return x.view(b_patch, c, self.patch_size, self.patch_size)

