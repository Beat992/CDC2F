import os
import random
import numpy as np
from validate import validate
from metrics.stream_metrics import StreamSegMetrics
from torch.utils.tensorboard import SummaryWriter
# from configs import WHU as cfg
from configs import CDD as cfg
# from configs import LEVIR_CD as cfg
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.CDDataset import CDDataset
from utils.logger import create_logger
from model.network import Network
from loss.loss import idx_guided_bce_loss, bce_withlogits
import torch
import torch.utils.data as Data


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train():
    # <-------create logger------->
    log = create_logger(logger_output_dir=os.path.join(cfg.base_path, 'log'),
                        net_name=cfg.model_name,
                        dataset_name=cfg.dataset_name,
                        phase='train')
    # <-------create tensorboard------>
    writer = SummaryWriter(log_dir='./log_tensorboard/{}/{}'.format(cfg.dataset_name, cfg.model_name))
    # <------------prepare dataset--------- >
    train_data = CDDataset(img_path=cfg.train_data_path,
                           label_path=cfg.train_label_path,
                           file_name_txt_path=cfg.train_txt_path,
                           split_flag='train',
                           img_size=256,
                           to_tensor=True, )

    val_data = CDDataset(img_path=cfg.val_data_path,
                         label_path=cfg.val_label_path,
                         file_name_txt_path=cfg.val_txt_path,
                         split_flag='val',
                         img_size=256,
                         to_tensor=True, )

    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=cfg.train_batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)

    val_loader = Data.DataLoader(dataset=val_data,
                                 batch_size=cfg.val_batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
    # <-------train configs------->
    start_epoch = 0
    num_epoch = cfg.num_epochs
    epoch_iters = len(train_data) // cfg.train_batch_size
    max_iters = cfg.num_epochs * epoch_iters
    best_metric = 0
    # <---------build  models---------->
    backbone = 'resnet18'
    stages_num = 4
    threshold = 0.5
    phase = 'train'
    encoder_depth = 4
    encoder_heads = 8
    encoder_dim = 16
    decoder_depth = 1
    decoder_heads = 8
    decoder_dim = 16
    dropout = 0.5
    dct_size = 4
    patch_size = 32

    log.info('network parameters: {}'.format(
        (backbone, stages_num, threshold, phase, dct_size, patch_size, encoder_depth, encoder_heads, encoder_dim,
         decoder_depth, decoder_heads, decoder_dim, dropout)))
    model = Network(backbone=backbone, stages_num=stages_num,
                    threshold=threshold, phase=phase, dct_size=dct_size, patch_size=patch_size,
                    encoder_depth=encoder_depth, encoder_heads=encoder_heads, encoder_dim=encoder_dim,
                    decoder_depth=decoder_depth, decoder_heads=decoder_heads, decoder_dim=decoder_dim,
                    dropout=dropout).cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # <---------- optimizer --------->
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.init_learning_rate, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10, verbose=True)
    # <-----resume------->
    if cfg.resume:
        checkpoints = torch.load(os.path.join(cfg.training_best_ckpt, cfg.save_name))
        model.load_state_dict(checkpoints['model_state'])
        optimizer.load_state_dict(checkpoints['optimizer_state'])
        scheduler.load_state_dict((checkpoints['scheduler_state']))
        start_epoch = checkpoints['cur_epoch']
        best_metric = checkpoints['best_score']
        log.info('resume from %s' % os.path.join(cfg.training_best_ckpt, cfg.save_name))

    # <--------- record configs -------->
    log.info('dataset: {}'.format(cfg.dataset_name))
    log.info('train set:{}'.format(len(train_data)))
    log.info('val set:{}'.format(len(val_data)))
    log.info(f'model: {cfg.model_name}')
    log.info('patch size:{}, dct size{}'.format(patch_size, dct_size))
    log.info('notes:{}'.format(cfg.notes))
    log.info('init learning rate:{}'.format(cfg.init_learning_rate))
    log.info(f'batch size: {cfg.train_batch_size}')
    log.info(f'epoch nums: {cfg.num_epochs}')
    log.info(f'threshold: {cfg.threshold}')

    # <--------- iter img-label pairs ---------->
    for epoch in range(start_epoch, num_epoch):
        model.train()
        start_event.record()
        for batch_idx, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            img1, img2, label = batch  # 数据
            img1 = img1.cuda()
            img2 = img2.cuda()
            label = label.cuda().reshape(-1, 1, 256, 256)

            coarse, coarse2, coarse3, fine, idx = model(img1, img2)
            coarse_loss = bce_withlogits(label, coarse)
            coarse2_loss = idx_guided_bce_loss(label, coarse2, idx, patch_size)
            coarse3_loss = bce_withlogits(label, coarse3)
            loss = coarse3_loss + coarse_loss + coarse2_loss

            writer.add_scalar(tag='fine patch num', scalar_value=idx.numel(), global_step=step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar(tag='coarse_loss', scalar_value=coarse_loss, global_step=step)
            writer.add_scalar(tag='coarse2_loss', scalar_value=coarse2_loss, global_step=step)
            writer.add_scalar(tag='coarse3_loss', scalar_value=coarse3_loss, global_step=step)

        metrics = StreamSegMetrics(n_classes=2)

        current_metric, P, R = validate(model, val_loader, metrics, log, epoch=epoch, writer=writer)
        scheduler.step(current_metric)
        writer.add_scalar(tag='F1', scalar_value=current_metric, global_step=epoch)

        if current_metric > best_metric:
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'cur_epoch': epoch,
                    'best_score': current_metric
                }, os.path.join(cfg.training_best_ckpt, cfg.save_name))
            best_metric = current_metric
            best_epoch = epoch
            log.info(
                'save_best_f1_pth_dir:{}\nbest_epoch:{}'.format(
                    os.path.join(cfg.training_best_ckpt, cfg.save_name), best_epoch))

        end_event.record()
        torch.cuda.synchronize()
        time = start_event.elapsed_time(end_event)
        log.info(f'一个epoch时间:{time}')
    writer.close()


def set_random_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed = 5555
    set_random_seeds(seed)
    train()
