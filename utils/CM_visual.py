import os.path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib import pyplot as plt
from model.FC.UNet_CD import Unet
from model.FC.fc_siam_conc import Siam_conc
from model.FC.fc_siam_di import Siam_diff
from model.P2VNet.p2v import P2VNet
from model.SARAS.model import SARAS
from model.STANet.stanet import STANet
from model.USSFC.USSFCNet import USSFCNet
from model.change_former.ChangeFormer import ChangeFormerV6
from model.help_function import size_restore
from model.network import Network
from model.BiT.networks import BASE_Transformer

def visual_change_map():
    base_path = ''
    model_dict = {
                  # 'UNet': Unet(6, 1),
                  # 'FC_conc': Siam_conc(3, 1),
                  # 'FC_diff': Siam_diff(3, 1),
                  # 'USSFC': USSFCNet(3, 1),
                  # 'STANet': STANet(3, 64, 'PAM'),
                  # 'P2V': P2VNet(3),
                  # 'ChangeFormer': ChangeFormerV6(output_nc=2, embed_dim=256),
                  # 'BiT': BASE_Transformer(input_nc=3, output_nc=1, token_len=4, resnet_stages_num=4,
                  #            with_pos='learned', enc_depth=1, dec_depth=8, dim_head=8, decoder_dim_head=8).cuda(),
                  'SARAS': SARAS()
                  }
    dataset_list = ['CDD']   #'LEVIR_CD','WHU','WHU'

    for dataset in dataset_list:
        if dataset == 'LEVIR_CD':
            img_list = ['0891']         #'0042''0046', '0640', '0891'' 0808', '0046'
            img_path = '/home/pan2/zwb/LEVIR-CD-256-NEW/test'
            img_type = '.png'
            label_name = 'label'
        elif dataset == 'WHU':
            img_list = ['2327', '3742', '4135']
            img_path = '/home/pan2/zwb/WHU/test'
            img_type = '.png'
            label_name = 'mask'
        else:
            img_list = ['00121', '00098', '00052', '00273', '00303', '00394', '00420', '00884', '01046', '01283', '01819', '01927', '02177']
            img_path = '/home/pan2/zwb/ChangeDetectionDataset/Real/subset/test'
            img_type = '.jpg'
            label_name = 'OUT'
        for model_name, model in model_dict.items():
            model = model.cuda()
            state_dict = torch.load(
                os.path.join(base_path, f'checkpoints_{dataset}/{dataset}_{model_name}_OK.pth'))
            model.load_state_dict(state_dict['model_state'])
            model.eval()


            for img in img_list:
                img1_path = img_path + '/A/' + img + img_type
                img2_path = img_path + '/B/' + img + img_type
                label_path = img_path + '/' + label_name + '/' + img + img_type

                img1 = Image.open(img1_path)
                img2 = Image.open(img2_path)
                if dataset == 'WHU':
                    label = np.asarray(Image.open(label_path))
                    label = np.where(label > 0, 255, 0)
                else:
                    label = np.asarray(Image.open(label_path))
                    # label = np.where(label > 0, 255, 0)
                img1 = TF.to_tensor(img1)
                img2 = TF.to_tensor(img2)

                img1 = TF.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).view(1, 3, 256, 256).cuda()
                img2 = TF.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).view(1, 3, 256, 256).cuda()

                # <-------以下为彩色CM可视化-------->
                prediction = 0
                if isinstance(model, Network):
                    _, _, _, prediction, _ = model(img1, img2)
                    prediction = torch.where(prediction > 0.5, 255, 0).squeeze()
                    prediction = prediction.detach().cpu()
                elif isinstance(model, P2VNet):
                    prediction, _ = model(img1, img2)
                    prediction = torch.where(prediction > 0.5, 255, 0).squeeze()
                    prediction = prediction.detach().cpu()
                elif isinstance(model, (Unet,
                                        Siam_conc,
                                        Siam_diff,)):
                    prediction = torch.sigmoid(model(img1, img2))
                    prediction = torch.where(prediction > 0.5, 255, 0).squeeze()
                    prediction = prediction.detach().cpu()
                elif isinstance(model, (USSFCNet,
                                        STANet,
                                        BASE_Transformer)):
                    prediction = model(img1, img2)
                    if isinstance(model, STANet):
                        t = 1.15
                    else:
                        t = 0.5
                    prediction = torch.where(prediction > t, 255, 0).squeeze()
                    prediction = prediction.detach().cpu()
                elif isinstance(model, SARAS):
                    prediction = model(img1, img2)
                    # prediction = torch.softmax(prediction, dim=1)
                    prediction = torch.argmax(prediction, 1, keepdim=True) * 255
                    prediction = prediction.detach().cpu().squeeze()
                    # prediction = F.interpolate(prediction.view(1, 1, 512, 512).float(), size=(256, 256), mode='nearest').detach().cpu().squeeze()

                elif isinstance(model, ChangeFormerV6):
                    prediction = model(img1, img2)
                    if prediction[4].shape[1] == 1:
                        prediction = (torch.sigmoid(prediction[4]) > 0.5) * 255
                    else:
                        prediction = torch.argmax(torch.softmax(prediction[4], dim=1), dim=1, keepdim=True) * 255
                    prediction = prediction.detach().cpu().squeeze()

                pred_array =  np.array(prediction)
                # Create an array for each condition
                true_positive = np.logical_and(pred_array == 255, label == 255)
                true_negative = np.logical_and(pred_array == 0, label == 0)
                false_positive = np.logical_and(pred_array == 255, label == 0)
                false_negative = np.logical_and(pred_array == 0, label == 255)

                # Create a result image using color codes
                result_image = np.zeros((*pred_array.shape, 3), dtype=np.uint8)
                result_image[true_positive] = (255, 255, 255)  # True Positive (White)
                result_image[true_negative] = (0, 0, 0)  # True Negative (Black)
                result_image[false_positive] = (255, 0, 0)  # False Positive (Red)
                result_image[false_negative] = (0, 255, 0)  # False Negative (Green)

                # Convert the result array to a PIL image
                result_image_pil = Image.fromarray(result_image)

                # Save the result image
                save_path = os.path.join(base_path, f'cm_visual_{dataset}/{model_name}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                result_image_pil.save(save_path+f'/{img}'+img_type)


if __name__ == '__main__':
    visual_change_map()