#encoding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
plt.rc('font', family='Times New Roman')
train_data = pd.read_csv('./stratification_train.csv')
#0:id, 1:label, 2:pred, 3:sex, 4:age, 5:BMI, 6:system, 7:thickness
# sex=female, sex=male
def get_sex_stati():
    indix_sex_female = np.flatnonzero(train_data["Sex"] == 0)
    sex_female_label = np.array(train_data.iloc[indix_sex_female, 1]).astype(int)
    sex_female_pred = np.array(train_data.iloc[indix_sex_female, 2]).astype(float)
    indix_sex_male = np.flatnonzero(train_data["Sex"] == 1)
    sex_male_label = np.array(train_data.iloc[indix_sex_male, 1]).astype(int)
    sex_male_pred = np.array(train_data.iloc[indix_sex_male, 2]).astype(float)
    female_male_label = np.array(train_data.iloc[:, 1]).astype(int)
    female_male_pred = np.array(train_data.iloc[:, 2]).astype(float)
    female_fpr, female_tpr, thresholds1 = roc_curve(sex_female_label, sex_female_pred)
    female_auc = auc(female_fpr, female_tpr)
    male_fpr, male_tpr, thresholds2 = roc_curve(sex_male_label, sex_male_pred)
    male_auc = auc(male_fpr, male_tpr)
    all_fpr, all_tpr, thresholds3 = roc_curve(female_male_label, female_male_pred)
    all_auc = auc(all_fpr, all_tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(female_fpr, female_tpr, label='Female(AUC = %0.3f)' % female_auc)
    ax.plot(male_fpr, male_tpr, label='Male(AUC = %0.3f)' % male_auc)
    ax.plot(all_fpr, all_tpr, label='Overall set(AUC = %0.3f)' % all_auc)

    ax.set_xlabel('1-Specificity',fontsize=12)
    ax.set_ylabel('Sensitivity',fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_title('ROC comparision')
    ax.legend(fontsize=12)
    plt.tight_layout()  # 调整布局
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.savefig('./img/sex_stratification.png', dpi=1000)  # 保存为文件
    plt.show()


def get_not_equal_stati(key, small, over, name1, name2, name3, save_name):
    indix_small = np.flatnonzero((train_data[key] < small) & (train_data[key].notnull()))
    small_label = np.array(train_data.iloc[indix_small, 1]).astype(int)
    small_pred = np.array(train_data.iloc[indix_small, 2]).astype(float)
    indix_over = np.flatnonzero((train_data[key] >= over) & (train_data[key].notnull()))
    over_label = np.array(train_data.iloc[indix_over, 1]).astype(int)
    over_pred = np.array(train_data.iloc[indix_over, 2]).astype(float)
    all_label = np.array(train_data.iloc[:, 1]).astype(int)
    all_pred = np.array(train_data.iloc[:, 2]).astype(float)

    small_fpr, small_tpr, thresholds1 = roc_curve(small_label, small_pred)
    small_auc = auc(small_fpr, small_tpr)
    over_fpr, over_tpr, thresholds2 = roc_curve(over_label, over_pred)
    over_auc = auc(over_fpr, over_tpr)
    all_fpr, all_tpr, thresholds3 = roc_curve(all_label, all_pred)
    all_auc = auc(all_fpr, all_tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(small_fpr, small_tpr, label=f'{name1}(AUC = %0.3f)' % small_auc)
    ax.plot(over_fpr, over_tpr, label=f'{name2}(AUC = %0.3f)' % over_auc)
    ax.plot(all_fpr, all_tpr, label=f'{name3}(AUC = %0.3f)' % all_auc)

    ax.set_xlabel('1-Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_title('ROC comparision')
    ax.legend(fontsize=12)
    plt.tight_layout()  # 调整布局
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.savefig(f'./img/{save_name}_stratification.png', dpi=1000)  # 保存为文件
    plt.show()
def get_system_stati():
    indix_sys_GE = np.flatnonzero(train_data["System"] == 1)
    sys_GE_label = np.array(train_data.iloc[indix_sys_GE, 1]).astype(int)
    sys_GE_pred = np.array(train_data.iloc[indix_sys_GE, 2]).astype(float)
    indix_sys_phi = np.flatnonzero(train_data["System"] == 2)
    sys_phi_label = np.array(train_data.iloc[indix_sys_phi, 1]).astype(int)
    sys_phi_pred = np.array(train_data.iloc[indix_sys_phi, 2]).astype(float)
    indix_sys_sim = np.flatnonzero(train_data["System"] == 3)
    sys_sim_label = np.array(train_data.iloc[indix_sys_sim, 1]).astype(int)
    sys_sim_pred = np.array(train_data.iloc[indix_sys_sim, 2]).astype(float)
    all_label = np.array(train_data.iloc[:, 1]).astype(int)
    all_pred = np.array(train_data.iloc[:, 2]).astype(float)

    GE_fpr, GE_tpr, thresholds1 = roc_curve(sys_GE_label, sys_GE_pred)
    GE_auc = auc(GE_fpr, GE_tpr)
    phi_fpr, phi_tpr, thresholds2 = roc_curve(sys_sim_label[:60], sys_sim_pred[:60])
    phi_auc = auc(phi_fpr, phi_tpr)
    sim_fpr, sim_tpr, thresholds3 = roc_curve(sys_sim_label, sys_sim_pred)
    sim_auc = auc(sim_fpr, sim_tpr)
    all_fpr, all_tpr, thresholds4 = roc_curve(all_label, all_pred)
    all_auc = auc(all_fpr, all_tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(GE_fpr, GE_tpr, label='GE CT system(AUC = %0.3f)' % GE_auc)
    ax.plot(phi_fpr, phi_tpr, label='Philips CT system(AUC = %0.3f)' % phi_auc)
    ax.plot(sim_fpr, sim_tpr, label='SIEMENS CT system(AUC = %0.3f)' % sim_auc)
    ax.plot(all_fpr, all_tpr, label='Overall set(AUC = %0.3f)' % all_auc)

    ax.set_xlabel('1-Specificity',fontsize=12)
    ax.set_ylabel('Sensitivity',fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_title('ROC comparision')
    ax.legend(fontsize=12)
    plt.tight_layout()  # 调整布局
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # plt.savefig('./img/system_stratification.png', dpi=1000)  # 保存为文件
    plt.show()
if __name__ == '__main__':
    # # sex
    # get_sex_stati()
    # # age
    # get_not_equal_stati('Age', 60, 60, 'Age < 60', "Age >= 60", "Overall set", 'age')
    # # BMI
    # get_not_equal_stati('BMI', 24, 24, 'BMI < 24', "BMI >= 24", "Overall set", 'BMI')
    # # thickness
    # get_not_equal_stati('Imagethickness', 2, 2, 'Image thickness < 5', "Image thickness >= 5", "Overall set", 'thickness')
    # # system
    get_system_stati()