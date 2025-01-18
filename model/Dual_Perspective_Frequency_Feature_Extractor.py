from math import sqrt
import torch
import torch.nn as nn
from model.layers import Transformer, TransformerDecoder

'''
    input: DCT patches which need to refinement, some of which is 0,
           means they are either change completely or no change 
    shape: (bs*16) * patch_n * 3*hf
    output: 
'''


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channel)
        y = self.excitation(y).view(batch_size, channel, 1, 1)
        return x * y


class DPFFE(nn.Module):
    def __init__(self,
                 dct_size=4,
                 patch_size=64,
                 encoder_depth=1,
                 encoder_heads=8,
                 encoder_dim=8,
                 decoder_depth=1,
                 decoder_heads=4,
                 decoder_dim=8,
                 dropout=0.5):
        super(DPFFE, self).__init__()
        self.dct_size = dct_size
        self.patch_size = patch_size
        self.mini_patch_num = (patch_size // dct_size) ** 2
        token_len_p = 3 * (dct_size**2)
        self.band_gconv = nn.Sequential(
            nn.Conv2d(in_channels=(dct_size**2)*3, out_channels=(dct_size**2)*3, kernel_size=3, stride=1, padding=1,
                      groups=(dct_size**2)*3),
        )
        self.band_seblock = SEBlock(channel=(dct_size ** 2) * 3, reduction=6)
        self.st2vector = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm2d((patch_size // dct_size) ** 2),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.BatchNorm2d((patch_size // dct_size) ** 2),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        self.patch2vetor = nn.Sequential(
            nn.Linear(dct_size ** 2, 64),
            nn.BatchNorm2d((patch_size // dct_size) ** 2),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.BatchNorm2d((patch_size // dct_size) ** 2),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.patch_attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm2d((patch_size // dct_size) ** 2),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.BatchNorm2d((patch_size // dct_size) ** 2),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.patch_embedding_layer = nn.Linear(token_len_p, token_len_p)
        self.patch_encoder = Transformer(dim=token_len_p,
                                              depth=encoder_depth,
                                              heads=encoder_heads,
                                              dim_head=encoder_dim,
                                              mlp_dim=token_len_p,
                                              dropout=dropout)
        self.patch_enc_pos_embedding = nn.Parameter(torch.randn(1, (patch_size // dct_size) ** 2, token_len_p))
        self.patch_dec_pos_embedding = nn.Parameter(torch.randn(1, (patch_size // dct_size) ** 2, token_len_p))
        self.patch_decoder = TransformerDecoder(dim=token_len_p, depth=decoder_depth, heads=decoder_heads,
                                            dim_head=decoder_dim, dropout=dropout, mlp_dim=decoder_dim)
        self.patch_dc_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        
    def _band_level(self, x):
        bs32, patch_num, ch, fre_num = x.shape
        x = x.transpose(1, 3).reshape(bs32, -1, int(sqrt(patch_num)), int(sqrt(patch_num)))
        x = self.band_gconv(x)
        x = self.band_seblock(x)
        x = x.reshape(bs32, -1, ch, patch_num).transpose(1, 3)
        return x

    def _patch_level(self, x1):
        bs16, patch_num, ch, fre_num = x1.shape
        x1 = x1.reshape(bs16, patch_num, -1)
        m1 = torch.clone(x1)
        x1 = self.embedding_layer_p2p(x1)
        x1 = x1 + self.enc_pos_embedding_p2p
        x1 = self.encoder_p(x1)
        x1 = self.decoder_p(x1, m1 + self.dec_pos_embedding_p2p)
        x1 = x1.view(bs16, patch_num, ch, -1)
        return x1

    def _patch_attn(self, x):
        bs16, patch_num, ch, fre_num = x.shape
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        min_vals = x.min(dim=-1, keepdim=True)[0]
        max_vals = x.max(dim=-1, keepdim=True)[0]
        peak_frequency = torch.argmax(x, dim=-1, keepdim=True)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        energy = torch.sum(torch.square(x), dim=-1, keepdim=True)
        st = torch.cat([var, mean, max_vals, min_vals, peak_frequency, rms, energy], dim=-1)
        st_v = self.st2vector(st)
        x_v = self.patch2vetor(x)
        attn = self.patch_attn(st_v + x_v).view(bs16, patch_num, 3, 1)
        x = attn * x
        x = x.reshape(bs16, patch_num, ch, fre_num)
        return x

    def forward(self, t0, t1, idx):
        # <------band level (intra patch)------>
        t0_bl = self._band_level(t0)
        t1_bl = self._band_level(t1)

        # <------patch level (inter patch)------>
        t0_pl = self._patch_attn(t0)
        t1_pl = self._patch_attn(t1)

        t0_pl = self._patch_level(t0_pl)
        t1_pl = self._patch_level(t1_pl)

        t0_dc_p = t0[:, :, :, 0].transpose(1, 2). \
            reshape(idx.numel(), 3, self.patch_size // self.dct_size, self.patch_size // self.dct_size)
        t1_dc_p = t1[:, :, :, 0].transpose(1, 2). \
            reshape(idx.numel(), 3, self.patch_size // self.dct_size, self.patch_size // self.dct_size)
        t0_dc_p = self.patch_dc_conv(t0_dc_p)
        t1_dc_p = self.patch_dc_conv(t1_dc_p)
        t0_dc_p = t0_dc_p.reshape(idx.numel(), 3, (self.patch_size // self.dct_size) ** 2).transpose(1, 2)
        t1_dc_p = t1_dc_p.reshape(idx.numel(), 3, (self.patch_size // self.dct_size) ** 2).transpose(1, 2)

        t0_pl[:, :, :, 0] = t0_pl[:, :, :, 0] + t0_dc_p
        t1_pl[:, :, :, 0] = t1_pl[:, :, :, 0] + t1_dc_p

        t0 = torch.cat([t0_bl, t0_pl], dim=2)
        t1 = torch.cat([t1_bl, t1_pl], dim=2)

        return t0, t1




