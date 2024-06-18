# encoding=utf-8
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

plt.rc('font', family='Times New Roman')
all_csv_name = ['train', 'val', 'test0', 'test1', 'test2']
train_label = []
train_Nomo = []
val_label = []
val_Nomo = []
test0_label = []
test0_Nomo = []
test1_label = []
test1_Nomo = []
test2_label = []
test2_Nomo = []
label_list = [train_label, val_label, test0_label, test1_label, test2_label]
Nomo_list = [train_Nomo, val_Nomo, test0_Nomo, test1_Nomo, test2_Nomo]
for i in range(len(all_csv_name)):
    for fold in range(1, 11):
        data = pd.read_csv(f'../../Utils/nomogram/nomogram_signature/fold_{fold}/{all_csv_name[i]}_data_{fold}.csv')
        y_true = np.array(data.iloc[:, 1]).astype(int)
        # nomogram
        y_pred6 = np.array(data.iloc[:, 6]).astype(float)
        label_list[i].extend(y_true)
        Nomo_list[i].extend(y_pred6)



fpr1, tpr1, thresholds1 = roc_curve(train_label, train_Nomo)
fpr2, tpr2, thresholds2 = roc_curve(val_label, val_Nomo)
fpr3, tpr3, thresholds3 = roc_curve(test0_label, test0_Nomo)
fpr4, tpr4, thresholds4 = roc_curve(test1_label, test1_Nomo)
fpr6, tpr6, thresholds6 = roc_curve(test2_label, test2_Nomo)
print(auc(fpr1, tpr1), auc(fpr2, tpr2), auc(fpr3, tpr3), auc(fpr4, tpr4), auc(fpr6, tpr6))
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr1, tpr1, label='Training cohort(AUC = %0.3f)' % auc(fpr1, tpr1))
ax.plot(fpr2, tpr2, label='Internal validation cohort(AUC = %0.3f)' % auc(fpr2, tpr2))
ax.plot(fpr3, tpr3, label='Internal test cohort (AUC = %0.3f)' % auc(fpr3, tpr3))
ax.plot(fpr4, tpr4, label='External validation cohort 1(AUC = %0.3f)' % auc(fpr4, tpr4))
ax.plot(fpr6, tpr6, label='External validation cohort 2(AUC = %0.3f)' % auc(fpr6, tpr6))
ax.set_xlabel('1-Specificity', fontsize=12)
ax.set_ylabel('Sensitivity', fontsize=12)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
# ax.set_title('ROC comparision')
ax.legend(fontsize=12)
plt.tight_layout()  # 调整布局
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.savefig('./img/roc_curves_nomogram.png', dpi=1000)  # 保存为文件
plt.show()
