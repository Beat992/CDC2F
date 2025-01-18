import torch
import torch.nn.functional as F
from torchjpeg import dct
from model.help_function import zigzag_extraction


class ToDCT:
    """
        input: origin img
        input size: bs, c, h, w
        output: each patch's dct coff
        output size: bs, patch num, patch len
    """
    def __init__(self, dct_size, patch_size=64):
        super(ToDCT, self).__init__()
        self.dct_size = dct_size
        self.patch_size = patch_size
        self.patch_num_large = (256//patch_size) ** 2
        self.patch_num_mini = (patch_size//dct_size) ** 2

    def forward(self, x1, x2, idx):
        x1 = self._blocky_(x1, self.patch_size)
        x2 = self._blocky_(x2, self.patch_size)
        x1 = x1.reshape(-1, 3, self.patch_size, self.patch_size)
        x2 = x2.reshape(-1, 3, self.patch_size, self.patch_size)
        x1 = torch.index_select(x1, dim=0, index=idx)
        x2 = torch.index_select(x2, dim=0, index=idx)
        x1 = self._to_dct(self._blocky_(x1, self.dct_size))
        x2 = self._to_dct(self._blocky_(x2, self.dct_size))

        return x1, x2

    @staticmethod
    def _blocky_(x, size):
        bs, ch, h, w = x.shape
        x = x.reshape(bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(size, size), dilation=1, padding=0, stride=(size, size))
        x = x.view(bs, ch, size, size, -1).permute(0, 4, 1, 2, 3)

        return x.reshape(bs, -1, ch, size, size)

    @staticmethod
    def _to_dct(x):
        # x = x + 1
        x = x / 2
        x = x + 0.5
        x = x * 255
        x = x - 128  # DCT requires that pixel value must in [-128, 127]
        x = x.transpose(1, 2)
        x = dct.block_dct(x)
        b, c, p, h, _ = x.shape
        x = x.view(-1, p, h, h)
        x = zigzag_extraction(x)
        x = x.reshape(b, c, p, -1).transpose(1, 2)
        return x
