#encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

from runner.Runner_Utils.evaluation_index import ClassificationMetric


def cal_auc_sens_spe_with95(y_true, y_score):
    # 设置bootstrap抽样次数
    n_bootstrap = 1000  # 您可以根据需要选择抽样次数

    # 初始化空数组以存储每次抽样的AUC、sens和spe
    bootstrap_aucs = []
    bootstrap_sens = []
    bootstrap_spe = []
    bootstrap_acc = []
    bootstrap_ppv = []
    bootstrap_npv = []

    for _ in range(n_bootstrap):
        # 随机抽样数据集（可以根据需要调整抽样大小）
        sampled_indices = resample(range(len(y_true)))
        y_true_sampled = y_true[sampled_indices]
        y_score_sampled = y_score[sampled_indices]
        metric = ClassificationMetric(2)  # 2表示有2个分类，有几个分类就填几
        # 计算AUC
        fpr, tpr, thresholds = roc_curve(y_true_sampled, y_score_sampled)
        auc_value = auc(fpr, tpr)
        bootstrap_aucs.append(auc_value)

        # 计算敏感性和特异性
        threshold = thresholds[np.argmax(tpr - fpr)]
        y_pred_sampled = (y_score_sampled > threshold).astype(int)
        hist = metric.addBatch(y_pred_sampled, y_true_sampled)
        acc_value = metric.accuracy()
        sens_value = metric.sensitivity()
        spe_value = metric.specificity()
        ppv_value = metric.ppv()
        npv_value = metric.npv()
        bootstrap_acc.append(acc_value)
        bootstrap_sens.append(sens_value)
        bootstrap_spe.append(spe_value)
        bootstrap_ppv.append(ppv_value)
        bootstrap_npv.append(npv_value)

    # 计算AUC、sens和spe的置信区间
    confidence_level = 0.95
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    conf_interval_auc = np.percentile(bootstrap_aucs, [lower_percentile, upper_percentile])
    conf_interval_acc = np.percentile(bootstrap_acc, [lower_percentile, upper_percentile])
    conf_interval_sens = np.percentile(bootstrap_sens, [lower_percentile, upper_percentile])
    conf_interval_spe = np.percentile(bootstrap_spe, [lower_percentile, upper_percentile])
    conf_interval_ppv = np.percentile(bootstrap_ppv, [lower_percentile, upper_percentile])
    conf_interval_npv = np.percentile(bootstrap_npv, [lower_percentile, upper_percentile])

    mean_auc = np.mean(bootstrap_aucs)
    mean_sens = np.mean(bootstrap_sens)
    mean_spe = np.mean(bootstrap_spe)
    mean_acc = np.mean(bootstrap_acc)
    mean_ppv = np.mean(bootstrap_ppv)
    mean_npv = np.mean(bootstrap_npv)

    print("95% AUC 置信区间: {:.3f} [{:.3f}, {:.3f}]".format(mean_auc, conf_interval_auc[0], conf_interval_auc[1]))
    print("95% Sensitivity 置信区间: {:.3f} [{:.3f}, {:.3f}]".format(mean_sens, conf_interval_sens[0],
                                                                     conf_interval_sens[1]))
    print("95% Specificity 置信区间: {:.3f} [{:.3f}, {:.3f}]".format(mean_spe, conf_interval_spe[0],
                                                                     conf_interval_spe[1]))
    print("95% Accuracy 置信区间: {:.3f} [{:.3f}, {:.3f}]".format(mean_acc, conf_interval_acc[0], conf_interval_acc[1]))
    print("95% PPV 置信区间: {:.3f} [{:.3f}, {:.3f}]".format(mean_ppv, conf_interval_ppv[0], conf_interval_ppv[1]))
    print("95% NPV 置信区间: {:.3f} [{:.3f}, {:.3f}]".format(mean_npv, conf_interval_npv[0], conf_interval_npv[1]))
all_csv_name = ['train','test0', 'test1', 'test2']
signature_name = ['Age', 'Sex', 'IPN signature', 'Adipose signature', "Age+Sex signature", 'Age+Sex+IPN signature', 'Nomogram']
for csv_name in all_csv_name:
    print(f'{csv_name}cohort的评价指标')
    data = pd.read_csv(f'./signature_8_csv/{csv_name}_data.csv')
    label = data.iloc[:, 1]
    j = 0
    for i in range(2, 9):
        print(signature_name[j])
        y_pred = data.iloc[:, i]
        cal_auc_sens_spe_with95(label, y_pred)
        j += 1