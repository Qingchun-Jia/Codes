import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
import os
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, auc

# 读取csv文件中的数据，不同的cohort，都读取最后一列的nomo预测值
train_data = pd.read_csv('../../Utils/nomogram/signature_8_csv/train_data.csv')
val_data = pd.read_csv('../../Utils/nomogram/signature_8_csv/val_data.csv')
test0_data = pd.read_csv('../../Utils/nomogram/signature_8_csv/test0_data.csv')
test1_data = pd.read_csv('../../Utils/nomogram/signature_8_csv/test1_data.csv')
test2_data = pd.read_csv('../../Utils/nomogram/signature_8_csv/test2_data.csv')

train_label = train_data.iloc[:, 1]
train_pred = train_data.iloc[:, 8]

val_label = val_data.iloc[:, 1]
val_pred = val_data.iloc[:, 8]

test0_label = test0_data.iloc[:, 1]
test0_pred = test0_data.iloc[:, 8]

test1_label = test1_data.iloc[:, 1]
test1_pred = test1_data.iloc[:, 8]

test2_label = test2_data.iloc[:, 1]
test2_pred = test2_data.iloc[:, 8]


# 定义SciPy通用颜色列表
colors = list(mcolors.TABLEAU_COLORS.keys())

# 绘制校准曲线
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# 分类器1的校准曲线
fraction_of_positives_1, mean_predicted_value_1 = calibration_curve(train_label, train_pred, n_bins=8, strategy='quantile')
ks_statistic_1, p_value_1 = ks_2samp(mean_predicted_value_1, fraction_of_positives_1)
plt.plot(mean_predicted_value_1, fraction_of_positives_1, "s-", color=colors[0], label=f"Training cohort")

# 分类器2的校准曲线
fraction_of_positives_2, mean_predicted_value_2 = calibration_curve(val_label, val_pred, n_bins=8, strategy='quantile')
ks_statistic_2, p_value_2 = ks_2samp(mean_predicted_value_2, fraction_of_positives_2)
plt.plot(mean_predicted_value_2, fraction_of_positives_2, "s-", color=colors[1], label=f"Internal validation cohort")

# 分类器3的校准曲线
fraction_of_positives_3, mean_predicted_value_3 = calibration_curve(test0_label, test0_pred, n_bins=8, strategy='quantile')
ks_statistic_3, p_value_3 = ks_2samp(mean_predicted_value_3, fraction_of_positives_3)
plt.plot(mean_predicted_value_3, fraction_of_positives_3, "s-", color=colors[2], label=f"Internal test cohort")

# 分类器5的校准曲线
fraction_of_positives_4, mean_predicted_value_4 = calibration_curve(test1_label, test1_pred, n_bins=8, strategy='quantile')
ks_statistic_4, p_value_4 = ks_2samp(mean_predicted_value_4, fraction_of_positives_4)
plt.plot(mean_predicted_value_4, fraction_of_positives_4, "s-", color=colors[3], label=f"External validation cohort 1")

# 分类器6的校准曲线
fraction_of_positives_5, mean_predicted_value_5 = calibration_curve(test2_label, test2_pred, n_bins=8, strategy='quantile')
ks_statistic_5, p_value_5 = ks_2samp(mean_predicted_value_5, fraction_of_positives_5)
plt.plot(mean_predicted_value_5, fraction_of_positives_5, "s-", color=colors[4], label=f"External validation cohort 2")


plt.legend(fontsize=12)
plt.xlabel("Nomogram predicted probability", fontsize=12)
plt.ylabel("Actual frequency", fontsize=12)
plt.title("")
plt.legend(loc="lower right")

# 保存图表为JPEG格式，分辨率为1000
output_file_path = './img/DC.png'
plt.savefig(output_file_path, dpi=1000)

plt.show()
