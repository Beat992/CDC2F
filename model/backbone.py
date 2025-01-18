import torch
import torch.nn as nn
from model.backbones import resnet


class ResNet(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 resnet_stages_num=4,):

        super(ResNet, self).__init__()
        self.resnet_stages_num = resnet_stages_num
        if backbone == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, False, False])
        elif backbone == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, False, True])
        elif backbone == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=False,
                                          replace_stride_with_dilation=[False, False, True])

    def forward(self, x):
        conv1_feature = self.resnet.conv1(x)
        conv1_feature = self.resnet.bn1(conv1_feature)
        conv1_feature = self.resnet.relu(conv1_feature)
        conv2_feature = self.resnet.maxpool(conv1_feature)  # 1/2, out_channel=64

        conv2_feature = self.resnet.layer1(conv2_feature)  # 1/4, in=64, out=64

        conv3_feature = self.resnet.layer2(conv2_feature)  # 1/8, in=64, out=128

        if self.resnet_stages_num == 4:
            conv4_feature = self.resnet.layer3(conv3_feature)  # 1/8, in=128, out=256
            return [conv1_feature, conv2_feature, conv3_feature, conv4_feature]

        elif self.resnet_stages_num == 5:
            conv4_feature = self.resnet.layer3(conv3_feature)  # 1/8, in=128, out=256
            conv5_feature = self.resnet.layer4(conv4_feature)
            return [conv1_feature, conv2_feature, conv3_feature, conv4_feature, conv5_feature]
        else:
            raise 'stages_num expected 4 or 5'
