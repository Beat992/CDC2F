import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from dataset.data_utils import CDDataAugmentation



class CDDataset(Dataset):

    def __init__(self,
                 img_path,
                 label_path,
                 file_name_txt_path,
                 split_flag,
                 img_size=256,
                 to_tensor=True,
                 ):

        self.label_path = label_path
        self.img_path = img_path
        self.img_txt_path = file_name_txt_path
        self.img_list = np.loadtxt(self.img_txt_path, dtype=str)
        self.flag = split_flag
        self.img_label_path_pairs = self.get_img_label_path_pairs()
        self.img_size = img_size
        self.to_tensor = to_tensor
        if self.flag == 'train':
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                # random_color_tf=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

    def get_img_label_path_pairs(self):
        img_label_pair_list = {}
        if self.flag == 'train' or 'val':
            for idx, did in enumerate(open(self.img_txt_path)):
                image1_name, image2_name, mask_name = did.strip("\n").split(' ')
                img1_file = os.path.join(self.img_path, image1_name)
                img2_file = os.path.join(self.img_path, image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file])

        return img_label_pair_list


    def __getitem__(self, index):

        img1_path, img2_path, label_path = self.img_label_path_pairs[index]
        ####### load images #############
        img1 = np.asarray(Image.open(img1_path).convert('RGB'))
        img2 = np.asarray(Image.open(img2_path).convert('RGB'))
        label = np.asarray(Image.open(label_path)).astype(np.float32)

        [img1, img2], [label] = self.augm.transform([img1, img2], [label], to_tensor=self.to_tensor)

        return img1, img2, label

    def __len__(self):

        return len(self.img_label_path_pairs)


