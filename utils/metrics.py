import numpy as np


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class RunningScore:

    def __init__(self, n_classes, monitored_metric):
        self.n_classes = n_classes
        self.monitored_metric = monitored_metric
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.scores = None

    def _fast_hist(self, label_pred, label_true, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2)
        return hist.reshape(n_class, n_class)

    def update(self, pred_labels, true_labels):
        if not isinstance(pred_labels, np.ndarray):
            pred_labels = to_numpy(pred_labels)
        if not isinstance(true_labels, np.ndarray):
            true_labels = to_numpy(true_labels)

        for lp, lt in zip(pred_labels, true_labels):
            self.confusion_matrix += self._fast_hist(lp.flatten(), lt.flatten(), self.n_classes)

    def get_scores(self):
        hist = self.confusion_matrix

        #      | pr_1  pr_0
        # gt_1 |  tp    fn
        # gt_0 |  fp    tn

        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp

        iou = tp / (tp + fn + fp)
        mean_iou = np.nanmean(iou)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)  # = tp / (tp + 0.5(fp + fn))
        mean_f1 = np.nanmean(f1)

        if self.monitored_metric == 'IoU':
            cls_score = dict(zip(range(self.n_classes), iou))
        elif self.monitored_metric == 'F1':
            cls_score = dict(zip(range(self.n_classes), f1))
        elif self.monitored_metric == 'Precision':
            cls_score = dict(zip(range(self.n_classes), precision))
        else:
            cls_score = None

        self.scores = {
            "Precision": np.nanmean(precision),
            "Recall": np.nanmean(recall),
            "F1": mean_f1,
            "IoU": mean_iou,
        }

        return self.scores, cls_score

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
