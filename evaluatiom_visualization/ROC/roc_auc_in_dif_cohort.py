# encoding=utf-8
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

plt.rc('font', family='Times New Roman')
train_data = pd.read_csv(f'../../Utils/nomogram/signature_8_csv/train_data.csv')
test0_data = pd.read_csv(f"../../Utils/nomogram/signature_8_csv/test0_data.csv")
test1_data = pd.read_csv(f"../../Utils/nomogram/signature_8_csv/test1_data.csv")
test2_data = pd.read_csv(f"../../Utils/nomogram/signature_8_csv/test2_data.csv")
all_csv_name = ['train', 'val', 'test0', 'test1', 'test2']
for i in range(len(all_csv_name)):
    csv_label = []
    csv_age = []
    csv_sex = []
    csv_IPN = []
    csv_Adi = []
    csv_Nomo = []
    for fold in range(1, 11):
        data = pd.read_csv(f'../../Utils/nomogram/nomogram_signature/fold_{fold}/{all_csv_name[i]}_data_{fold}.csv')
        y_true = np.array(data.iloc[:, 1]).astype(int)
        # age
        y_pred1 = np.array(data.iloc[:, 5]).astype(int)
        # sex
        y_pred2 = np.array(data.iloc[:, 4]).astype(int)
        # IPN
        y_pred3 = np.array(data.iloc[:, 2]).astype(float)
        # Adipose
        y_pred4 = np.array(data.iloc[:, 3]).astype(float)
        # nomogram
        y_pred6 = np.array(data.iloc[:, 6]).astype(float)

        csv_label.extend(y_true)
        csv_age.extend(y_pred1)
        csv_sex.extend(y_pred2)
        csv_IPN.extend(y_pred3)
        csv_Adi.extend(y_pred4)
        csv_Nomo.extend(y_pred6)
    for j in range(len(csv_sex)):
        if csv_sex[j] == 0:
            csv_sex[j]=1
        else:
            csv_sex[j]=0
    fpr1, tpr1, thresholds1 = roc_curve(csv_label, csv_age)
    fpr2, tpr2, thresholds2 = roc_curve(csv_label, csv_sex)
    fpr3, tpr3, thresholds3 = roc_curve(csv_label, csv_IPN)
    fpr4, tpr4, thresholds4 = roc_curve(csv_label, csv_Adi)
    fpr6, tpr6, thresholds6 = roc_curve(csv_label, csv_Nomo)
    print(auc(fpr1, tpr1), auc(fpr2, tpr2), auc(fpr3, tpr3), auc(fpr4, tpr4), auc(fpr6, tpr6))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr1, tpr1, label='Age(AUC = %0.3f)' % auc(fpr1, tpr1))
    ax.plot(fpr2, tpr2, label='Gender(AUC = %0.3f)' % auc(fpr2, tpr2))
    ax.plot(fpr3, tpr3, label='IPN signature(AUC = %0.3f)' % auc(fpr3, tpr3))
    ax.plot(fpr4, tpr4, label='Adipose signature(AUC = %0.3f)' % auc(fpr4, tpr4))
    ax.plot(fpr6, tpr6, label='Nomogram(AUC = %0.3f)' % auc(fpr6, tpr6))
    ax.set_xlabel('1-Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_title('ROC comparision')
    ax.legend(fontsize=12)
    plt.tight_layout()  # 调整布局
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.savefig(f'./img/roc_curves_nomogram_{all_csv_name[i]}.png', dpi=1000)  # 保存为文件
    plt.show()


