import numpy as np
from sklearn.metrics import confusion_matrix


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)
            else:
                for i in range(len(v)):
                    string += "%s_%d: %f\n" % (k, i, v[i])
                    # string += "%s_%d\n"%(k,i)
        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IoU
            - fwavacc

        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)  # 每一类的precision
        acc_cls = np.nanmean(acc_cls)  # 忽略nan然后计算平均值
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # 每一类的IoU，[阴性， 阳性] —> [未变化， 变化]
        mean_iou = np.nanmean(iou)
        freq = hist.sum(axis=1) / hist.sum()  # 预测出每一类出现的概率
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        # cls_iu = dict(zip(range(self.n_classes), iu))
        cls_iou = []
        for i in range(len(iou)):
            iou_i = iou[i]
            cls_iou.append(iou_i)
        cls_iou = np.array(cls_iou)
        return {
            "Overall Acc": acc,          # 反映两类整体的准确率
            "Mean Acc": acc_cls,         # 各自准确率的平均值
            "FreqW Acc": fwavacc,        # 各自准确率的加权平均值，权重为频率
            "Mean IoU": mean_iou,        # 各自IoU的平均值
            "Class IoU": cls_iou,        # 两类各自的IoU
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
