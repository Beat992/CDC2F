import torch
import torch.nn.functional as F
from model.help_function import size_restore
from model.help_function import zigzag_extraction


def bce(gt, x):
    x = x.flatten()
    gt = torch.where(gt != 0, 1, 0).flatten().float()
    loss = F.binary_cross_entropy(input=x, target=gt, reduction='mean')
    return loss

def bce_withlogits(gt, x):
    x = x.flatten()
    gt = torch.where(gt != 0, 1, 0).flatten().float()
    loss = F.binary_cross_entropy_with_logits(input=x, target=gt, reduction='mean')
    return loss


def balanced_bce(gt, x):
    bs, _, _, _ = x.shape
    gt_ = (gt > 0).float().view(bs, -1)
    x = x.squeeze().view(bs, -1)

    pos_index = (gt_ == 1)
    neg_index = (gt_ == 0)
    ignore_index = (gt_ > 1)

    gt_[pos_index] = 1
    gt_[neg_index] = 0

    weight = torch.Tensor(x.size()).fill_(0)

    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum = pos_num + neg_num

    weight[pos_index] = neg_num * 1.0 / (sum + 1e-9)
    weight[neg_index] = pos_num * 1.0 / (sum + 1e-9)
    weight[pos_index] = 0.7
    weight[neg_index] = 0.3
    weight[ignore_index] = 0

    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(input=x, target=gt_, weight=weight, reduction='mean')
    return loss


def idx_guided_bce_loss(gt, x, idx, patch_size):
    bs, c, h, _ = gt.shape
    gt = torch.where(gt != 0, 1, 0).float().view(bs*c, 1, h, -1)
    gt = F.unfold(gt, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), dilation=1, padding=0)  # bs, 4096, patch_n
    gt = gt.view(bs, patch_size, patch_size, -1).permute(0, 3, 1, 2).reshape(-1, 1, patch_size, patch_size)
    gt = torch.index_select(gt, 0, idx)

    gt = gt.flatten()
    x = x.flatten()

    return F.binary_cross_entropy_with_logits(input=x, target=gt, reduction='mean')



