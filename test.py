import os.path
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from thop import profile
from tqdm import tqdm
from dataset.CDDataset import CDDataset
from metrics import StreamSegMetrics, metric_SeK
from model.FC.fc_siam_conc import Siam_conc
from model.FC.UNet_CD import Unet
from model.FC.fc_siam_di import Siam_diff
from model.BiT.networks import BASE_Transformer
from model.P2VNet.p2v import P2VNet
from model.STANet.stanet import STANet
from model.USSFC.USSFCNet import USSFCNet
from model.SARAS.model import SARAS
from model.network import Network
from utils.logger import create_logger
# import configs.LEVIR_CD as cfg
# import configs.WHU as cfg
import configs.CDD as cfg
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def test(threshold):
    log = create_logger(logger_output_dir=os.path.join(cfg.base_path, 'log'),
                        net_name=cfg.model_name,
                        dataset_name=cfg.dataset_name,
                        phase='test')
    model = Network(backbone='resnet18', stages_num=4,
                    threshold=0.5, phase='test', dct_size=4, patch_size=32,
                    encoder_depth=4, encoder_heads=8, encoder_dim=16,
                    decoder_depth=1, decoder_heads=8, decoder_dim=16,
                    dropout=0.5).cuda()
    # model = BASE_Transformer(input_nc=3, output_nc=1, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=1, dec_depth=8, dim_head=8, decoder_dim_head=8).cuda()
    # model = Unet(6, 1).cuda()
    # model = Siam_diff(3, 1).cuda()
    # model = Siam_conc(3, 1).cuda()
    # model = SARAS().cuda()
    # model = P2VNet(3).cuda()
    # model = USSFCNet(3, 1).cuda()
    # model = ChangeFormerV6().cuda()
    # model = STANet(3, 64, 'PAM').cuda()
    state_dict = torch.load('')
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
                                      num_workers=2,
                                      pin_memory=True)
    metrics = StreamSegMetrics(n_classes=2)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    pred_metric_all = []
    gt_metric_all = []
    metrics.reset()
    model.eval()
    totle_flops = 0
    blocks = 0
    with torch.no_grad():
        start_event.record()
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            inputs1, input2, mask = batch
            inputs1, inputs2, mask = inputs1.cuda(), input2.cuda(), mask.cuda()
            # coarse, coarse2, coarse3, fine, idx = model(inputs1, inputs2)
            if isinstance(model, (Unet,
                                  Siam_conc,
                                  Siam_diff,)):
                fine = model(inputs1, inputs2)
                fine = torch.sigmoid(fine)
                fine = fine.detach().cpu().numpy()
                pred_cm = np.uint8(np.where(fine > threshold, 1, 0)).squeeze()
                gt_cm = np.uint8(mask.squeeze().cpu().numpy())
                gt_cm = np.where(gt_cm > 0, 1, 0)
            elif isinstance(model, SARAS):
                fine = model(inputs1, inputs2)
                # fine = torch.softmax(fine, dim=1)
                pred_cm = torch.argmax(fine, 1).cpu().numpy().squeeze()
                if cfg.dataset_name != 'CDD':
                    mask = F.interpolate(mask, 512, mode='nearest')
                gt_cm = np.uint8(mask.squeeze().cpu().numpy())
                gt_cm = np.where(gt_cm > 0, 1, 0)
            elif isinstance(model, (USSFCNet,
                                    STANet,
                                    BASE_Transformer)):
                fine = model(inputs1, inputs2)
                fine = fine.squeeze().detach().cpu().numpy()
                if isinstance(model, STANet):
                    threshold = 1.05

                pred_cm = np.uint8(np.where(fine > threshold, 1, 0))
                gt_cm = np.uint8(mask.squeeze().cpu().numpy())
                gt_cm = np.where(gt_cm > 0, 1, 0)
            elif isinstance(model, Network):
                flops, _ = profile(model, inputs=(inputs1, inputs2), verbose=False)
                totle_flops = totle_flops + flops
                coarse1, coarse2, coarse3, fine, fine_idx = model(inputs1, inputs2)
                blocks = blocks + torch.nonzero(fine_idx).numel()
                fine = fine.detach().cpu().numpy()
                pred_cm = np.uint8(np.where(fine > threshold, 1, 0)).squeeze()
                gt_cm = np.uint8(mask.squeeze().cpu().numpy())
                gt_cm = np.where(gt_cm > 0, 1, 0)
            elif isinstance(model, P2VNet):
                fine, _ = model(inputs1, inputs2)
                fine = fine.detach().cpu().numpy()
                pred_cm = np.uint8(np.where(fine > threshold, 1, 0)).squeeze()
                gt_cm = np.uint8(mask.squeeze().cpu().numpy())
                gt_cm = np.where(gt_cm > 0, 1, 0)
            pred_metric_all.append(pred_cm)
            gt_metric_all.append(gt_cm)
            metrics.update(gt_cm, pred_cm)  # 添加

        end_event.record()
        torch.cuda.synchronize()
        time = start_event.elapsed_time(end_event)
        print(f'一个epoch:{time}')
        print(blocks)
    # <-----返回F1、P、R、MIoU>
    F1, P, R = metric_SeK(infer_array=np.array(pred_metric_all), label_array=np.array(gt_metric_all), n_class=2, log=log)

    print(f"模型的FLOPs: {totle_flops / 1e12} T FLOPs")
    print(f'F1 Score : {F1}, \nPrecision : {P}, \nRecall : {R}')
    score = metrics.get_results()  # 添加
    log.info('metric: ')
    log.info(metrics.to_str(score))

    return F1

if __name__ == '__main__':
    threshold = 0.5
    f1 = test(threshold)