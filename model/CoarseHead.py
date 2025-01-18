import torch
import torch.nn as nn
import torch.nn.functional as F


class Coarse(nn.Module):
    def __init__(self,
                 channel,
                 threshold,
                 patch_size,
                 stages_num=5,
                 phase='train'):
        super(Coarse, self).__init__()
        self.patch_size = patch_size
        self.stages_num = stages_num
        self.threshold = threshold
        self.patch_num = (256//patch_size)**2
        self.fine_index = None
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=256, kernel_size=1, stride=1, padding=0, ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, ),
        )
        self.pred_head1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1, ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, feature):
        size = feature[0][0].shape[-1]
        for item in feature:
            for i in range(self.stages_num):
                item[self.stages_num - 1 - i] = F.interpolate(item[self.stages_num - 1 - i], [size, size],
                                                              mode='bilinear', align_corners=True)
                if i != 0:
                    item[self.stages_num - 1 - i] = torch.cat([item[self.stages_num - 1 - i], item[self.stages_num - i]]
                                                              , dim=1)
                    del item[self.stages_num - i]
        feature[0] = self.coarse_conv(feature[0][0])
        feature[1] = self.coarse_conv(feature[1][0])
        coarse_score = torch.abs(feature[0] - feature[1])
        coarse_score = self.pred_head1(coarse_score)
        coarse_prob = torch.sigmoid(coarse_score)   # bs * 1 * 256 * 256
        coarse_mask = (coarse_prob > self.threshold).float()

        if self.phase == 'train':      # training with all the blocks
            self.fine_index = torch.ones(coarse_mask.shape[0]*(256//self.patch_size)**2).cuda()
        elif self.phase == 'val' or 'test':  # filtering during validation or test
            self.fine_index = F.unfold(coarse_mask, kernel_size=(self.patch_size, self.patch_size),
                                  padding=0, stride=(self.patch_size, self.patch_size))  # bs * 4096 * 16
            self.fine_index = (self.fine_index.transpose(1, 2).sum(dim=2)).flatten()
            self.fine_index = torch.gt(self.fine_index, 0) & torch.lt(self.fine_index, self.patch_size ** 2)
        # print(self.fine_index.sum())
        spa_feature = torch.cat([feature[0], feature[1]], dim=1)

        return coarse_score, torch.nonzero(self.fine_index).squeeze(), spa_feature





