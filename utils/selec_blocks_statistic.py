import os.path
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from thop import profile
from tqdm import tqdm
from dataset.LEVIR_CD import CDDataset
from model.network import Network
from utils.logger import create_logger
import configs.LEVIR_CD as cfg
# import configs.WHU as cfg
# import configs.CDD as cfg

def statictic():
    model = Network(backbone='resnet18', stages_num=4,
                    threshold=0.5, phase='test', dct_size=4, patch_size=8,
                    encoder_depth=4, encoder_heads=8, encoder_dim=16,
                    decoder_depth=1, decoder_heads=8, decoder_dim=16,
                    dropout=0.5).cuda()
    state_dict = torch.load()
    model.load_state_dict(state_dict['model_state'])

    test_data = CDDataset(img_path=cfg.test_data_path,
                          label_path=cfg.test_label_path,
                          file_name_txt_path=cfg.test_txt_path,
                          split_flag='test',
                          img_size=256,
                          to_tensor=True, )

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=cfg.val_batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      pin_memory=True)

    patch_size = 128
    blocks = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            inputs1, inputs2, mask = batch
            inputs1, inputs2, mask = inputs1.cuda(), inputs2.cuda(), mask.cuda()
            # _, _, _, _, fine_index = model(inputs1, inputs2)
            mask = torch.where(mask > 0, 1., 0.)
            fine_index = F.unfold(mask, kernel_size=(patch_size, patch_size),
                                  padding=0, stride=(patch_size, patch_size))  # bs * 4096 * 16
            fine_index = (fine_index.transpose(1, 2).sum(dim=2)).flatten()
            fine_index = torch.gt(fine_index, 0) & torch.lt(fine_index, patch_size ** 2)
            blocks = blocks + torch.sum(fine_index)
            # blocks = blocks + torch.nonzero(fine_index).numel()

    print(blocks)

if __name__ == '__main__':
    statictic()