import numpy as np

num_class = 2


def fast_hist(mask, pred_mask, num_class):
    index = (mask >= 0) & (mask < num_class)
    return np.bincount(num_class * mask[index].astype(int) + pred_mask[index], minlength=num_class ** 2).reshape(
        num_class, num_class)


def get_hist(pred_mask, mask):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(mask.flatten(), pred_mask.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()  # 对角线之和/all之和
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2  # 一行*一列 / all之和^2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
            print('po : {} pe : {}'.format(po, pe))
    return kappa


def metric_SeK(infer_array, label_array, n_class, log=None):
    # 这个代码很多语义类别的时候，miou不适用，因为这里的miou仅仅计算的时候两类情况下的
    hist = np.zeros((n_class, n_class))

    hist += get_hist(infer_array, label_array)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    IoU_mean = (iou[0] + iou[1]) / 2

    TN = hist[0][0]
    FP = hist[0][1]
    FN = hist[1][0]
    TP = hist[1][1]
    P = TP / ((TP + FP) + 1e-8)  # 认为变化是正例
    R = TP / ((TP + FN) + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)

    if log is not None:
        log.info('TP : {} TN : {} FP : {} FN : {}'.format(hist[1][1], hist[0][0], hist[0][1], hist[1][0]))
        log.info('Mean IoU = %.6f' % IoU_mean)
        log.info('P=%.6f' % P)
        log.info('R=%.6f' % R)
        log.info('F1=%.6f' % F1)

    return F1, P, R
