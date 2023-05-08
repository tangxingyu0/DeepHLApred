import numpy as np


def sn_sp_acc_mcc(true_label, predict_label, pos_label=1):
    import math
    pos_num = np.sum(true_label == pos_label)
    print('pos_num=', pos_num)
    neg_num = true_label.shape[0] - pos_num
    print('neg_num=', neg_num)
    tp = np.sum((true_label == pos_label) & (predict_label == pos_label))
    print('tp=', tp)
    tn = np.sum(true_label == predict_label) - tp
    print('tn=', tn)
    sn = tp / pos_num
    sp = tn / neg_num
    acc = (tp + tn) / (pos_num + neg_num)
    fn = pos_num - tp
    fp = neg_num - tn
    print('fn=', fn)
    print('fp=', fp)

    tp = np.array(tp, dtype=np.float64)
    tn = np.array(tn, dtype=np.float64)
    fp = np.array(fp, dtype=np.float64)
    fn = np.array(fn, dtype=np.float64)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)))
    return sn, sp, acc, mcc