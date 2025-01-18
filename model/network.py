import torch
import torch.nn as nn
import torch.nn.functional as F
from model.help_function import size_restore
from model.backbone import ResNet
from model.CoarseHead import Coarse
from model.to_dct import ToDCT
from model.Dual_Perspective_Frequency_Feature_Extractor import DPFFE
from model.Frequency_Domain_Information_Exchange import FIM
from model.Fre_Feature_Pred_Head import FreFeaturePred
from model.Spa_Fre_Feature_Fusion import SpaFreInteractionFusionModule
from configs.LEVIR_CD import pretrain_model_path_ResNet18

class Network(nn.Module):


    def __init__(self,
                 backbone='resnet18', stages_num=4,
                 threshold=0.5, phase='val', dct_size=4, patch_size=32,
                 encoder_depth=1, encoder_heads=8, encoder_dim=8,
                 decoder_depth=1, decoder_heads=8, decoder_dim=8,
                 dropout=0.5):
        super(Network, self).__init__()
        if stages_num == 4:
            if backbone == 'resnet18':
                self.channel = 512
            else:
                self.channel = 1856   # resnet50
        elif stages_num == 5:
            if backbone == 'resnet18':
                self.channel = 1024
            else:
                self.channel = 3904
        self.dct_size = dct_size
        self.patch_size = patch_size
        self.patch_num_large = (256 // patch_size) ** 2
        self.phase = phase
        self.backbone = ResNet(backbone=backbone, resnet_stages_num=stages_num)
        self.coarse = Coarse(stages_num=stages_num, threshold=threshold, patch_size=patch_size, channel=self.channel,
                             phase=phase)
        self.to_dct = ToDCT(dct_size=dct_size, patch_size=patch_size)
        self.dpffe = DPFFE(dct_size=dct_size, patch_size=patch_size,
                           encoder_depth=encoder_depth, encoder_heads=encoder_heads, encoder_dim=encoder_dim,
                           decoder_depth=decoder_depth, decoder_heads=decoder_heads, decoder_dim=decoder_dim,
                           dropout=dropout)
        self.fim = FIM(dct_size=dct_size, patch_size=32)
        self.frepred = FreFeaturePred(dct_size=dct_size, patch_size=patch_size)
        self.sfifm = SpaFreInteractionFusionModule(patch_size=patch_size)
        self.pred_head2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

        # Experiments show that fusing two probability maps with convolutional layers is not as effective as adding them
        # together directly

        # self.fine_tune = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=32, kernel_size=1, stride=1, padding=0, padding_mode='replicate'),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, ), )
        pretrain_dict = torch.load(pretrain_model_path_ResNet18)
        self.backbone.resnet.load_state_dict(pretrain_dict)

    def forward(self, x1, x2):
        b = int(x1.shape[0])


        #<----------- coarse detection ---------->
        spa_feature = [self.backbone(x1), self.backbone(x2)]
        coarse_score, fine_idx, spa_feature, finetune_mask = self.coarse(spa_feature)
        if fine_idx.numel() == 0:     # during valiation or test, fine_idx may be 0, which means coarse change map are all 1 or 0
            coarse2_score = 0
            coarse3_score = 0
            fine = torch.sigmoid(coarse_score)
            return coarse_score, coarse2_score, coarse3_score, fine, fine_idx
        coarse_prob_sel = F.unfold(coarse_score, kernel_size=(self.patch_size, self.patch_size),
                                          padding=0, stride=(self.patch_size, self.patch_size))
        coarse_prob_sel = coarse_prob_sel.transpose(1, 2).reshape(-1, self.patch_size, self.patch_size)
        coarse_prob_sel = torch.index_select(coarse_prob_sel, dim=0, index=fine_idx.squeeze())
        #<----------- frequency domain cd stream ---------->
        x1, x2 = self.to_dct(x1, x2, fine_idx)
        x1, x2 = self.dpffe(x1, x2, fine_idx)
        x1, x2 = self.fim(x1, x2)
        coarse3_score, fre_feature = self.frepred(x1, x2)  # bs16, 1, 64, 64
        coarse3_score = coarse3_score.view(b, -1, self.patch_size**2)
        coarse3_score = size_restore(coarse3_score, input_size=self.patch_size, output_size=256)
        t0, t1 = self.sfifm(spa_feature, fre_feature, coarse_prob_sel, fine_idx)
        # <-----------fine detection--------->
        coarse2_score = torch.abs(t0 - t1)
        coarse2_score = self.pred_head2(coarse2_score)
        coarse2_score_restored = torch.zeros(b*self.patch_num_large, 1, self.patch_size, self.patch_size).to(coarse2_score.dtype).cuda()
        coarse2_score_restored[fine_idx, :, :, :] = coarse2_score
        coarse2_score = coarse2_score_restored.view(b, -1, self.patch_size**2)
        coarse2_score = size_restore(coarse2_score, input_size=self.patch_size, output_size=256)
        fine = torch.sigmoid(coarse_score + coarse2_score)

        return coarse_score, coarse2_score, coarse3_score, fine, fine_idx

if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile

    A = torch.randn(1, 3, 256, 256).cuda()
    B = torch.randn(1, 3, 256, 256).cuda()
    model = Network('resnet18', 4, 0.5, 'train', 4, 32, 4, 8, 16, 1, 8, 16, 0.5).cuda()
    summary(model=model, input_size=[(3, 256, 256), (3, 25, 256)], batch_size=1, device="cuda")
    flops, params = profile(model, inputs=(A, B))
    print(f"模型的FLOPs: {flops / 1e9} G FLOPs")  # 以十亿FLOPs为单位显示
    print(f"模型的参数数量: {params / 1e6} M")
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))