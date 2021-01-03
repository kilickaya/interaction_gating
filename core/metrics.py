import numpy as np
from sklearn.metrics import average_precision_score

# region mAP

def mean_avg_precision_charades(y_true, y_pred):
    """ Returns mAP """
    m_aps = []
    n_classes = y_pred.shape[1]
    for oc_i in range(n_classes):
        pred_row = y_pred[:, oc_i]
        sorted_idxs = np.argsort(-pred_row)
        true_row = y_true[:, oc_i]
        tp = true_row[sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(y_pred.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    return m_ap

def mean_avg_precision_sklearn(y_true, y_pred):
    # """ Returns mAP """
    n_classes = y_true.shape[1]
    map = [average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]
    map = np.nan_to_num(map)
    map = np.mean(map)
    return map

# endregion

# region Accuracy

def calc_top_n_score(n_top, y, y_cap):
    n_corrects = 0
    for gt, pr in zip(y, y_cap):
        idx = np.argsort(pr)[::-1]
        idx = idx[0:n_top]
        gt = np.where(gt == 1)[0][0]
        if gt in idx:
            n_corrects += 1
    n = len(y)
    score = n_corrects / float(n)
    return score

# endregion