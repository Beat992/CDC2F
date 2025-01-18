import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchjpeg import dct
from model.help_function import zigzag_extraction, inverse_zigzag
import matplotlib.pyplot as plt

def images_dct_resolve(x, jiezhi, dct_size):
    bs, ch, h, w = x.shape
    patch_num = (h//dct_size) ** 2
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(dct_size, dct_size), padding=0, stride=(dct_size, dct_size))  # original image blocky
    x = x.transpose(1, 2).reshape(bs, ch, patch_num, dct_size, dct_size)
    x = dct.block_dct(x)  # b * 3 * 64 * 8 * 8
    b, c, p, h, _ = x.shape
    x = x.view(-1, p, h, h)
    x = zigzag_extraction(x).view(-1, p, h, h)
    # mask = torch.zeros_like(x)
    # mask[:, :, jiezhi-1] = 1
    mask = create_mask(h, jiezhi).view(1, 1, h, h)
    x = x * mask
    x = x.view(-1, p, h**2)
    x = inverse_zigzag(x, dct_size)
    x = dct.block_idct(x)
    x = x.view(bs*ch, patch_num, -1).transpose(1, 2)
    x = F.fold(x, kernel_size=(dct_size, dct_size), output_size=(256, 256), padding=0, stride=(dct_size, dct_size))
    return x

def create_mask(size, rounds):
    if size <= 0 or rounds <= 0:
        raise ValueError("Size and rounds should be positive integers.")

    mask = torch.zeros((size, size), dtype=bool)

    if rounds < size:
        for i in range(size):
            if rounds <= 0:
                pass
            else:
                mask[i, 0:rounds] = True
                rounds = rounds - 1
    else:
        for r in range(rounds):
            for i in range(min(r + 1, size)):
                if (r - i) < size and i < size:  # Check if the indices are within bounds
                    mask[r - i, i] = True

        # Keep one column from the rightmost diagonal
        for i in range(rounds, size):
            mask[:, i] = True

    return mask

if __name__ == '__main__':
    img1_path = ''  # 0886.png, 720 367 353 281 262 0811
    img2_path = ''

    img1_or = Image.open(img1_path)
    img2_or = Image.open(img2_path)
    img1 = np.array(img1_or).transpose([2, 0, 1])
    img2 = np.array(img2_or).transpose([2, 0, 1])
    img1 = torch.from_numpy(img1).unsqueeze(0).float()
    img2 = torch.from_numpy(img2).unsqueeze(0).float()
    dct_size = 16
    for i in range(1,dct_size**2):
        img_dct = images_dct_resolve(img1, i, dct_size)
        img_dct = img_dct[0].squeeze().numpy().astype(np.uint8)
        plt.imshow(img_dct, cmap='viridis')
        plt.axis('off')
        # plt.show()
        plt.savefig(f'dct/dct_{i}.png', bbox_inches='tight', pad_inches=0)