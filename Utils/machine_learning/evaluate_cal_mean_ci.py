import numpy as np


def cal_mean_CI(list):
    # ����ȫ�ֵĴ�ӡѡ�������λС��
    np.set_printoptions(precision=3)
    # �����ֵ
    mean = np.mean(list)
    # ʮ�۽�����֤�������ɶ���10
    n = 10
    # �����׼��
    std = np.std(list)
    # ˫��alpha0.05 = 2.228
    a = 2.228
    mean = np.round(mean, 3)
    ci_low = np.round(mean - (a * std / np.sqrt(n)), 3)
    ci_high = np.round(mean + (a * std/np.sqrt(n)), 3)
    return mean, [ci_low, ci_high]
def cal_acc_sens_spec_ppv_npv(threshold, y_true, y_pred):
    for j in range(len(y_pred)):
        if y_pred[j] > threshold:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
    y_pred_argmax = y_pred
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pred_argmax)):
        if y_pred_argmax[i] == y_true[i]:
            if y_true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn + 0.00001)
    spec = tn / (tn + fp + 0.00001)
    ppv = tp / (tp + fp + 0.00001)
    npv = tn / (tn + fn + 0.00001)
    return acc, sens, spec, ppv, npv