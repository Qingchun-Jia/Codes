#encoding=utf-8
import os
import sys

from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
# from models.swin_transfomer import swin_small_patch4_window7_224
import os
import sys
from Utils import get_ten_fold_dataset
from models.cla_models import swin_small_patch4_window7_224

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
def cal_mean_CI(list):
    # 计算均值
    mean = np.mean(list)
    # 十折交叉验证所以自由度是10
    n = 10
    # 计算标准差
    std = np.std(list)
    # 双边alpha0.05 = 2.228
    a = 2.228

    return mean, [mean - (a * std/np.sqrt(n)), mean + (a * std/np.sqrt(n))]
def cal_acc_sens_spec_ppv_npv(threshold, y_true, y_pred):
    for j in range(len(y_pred)):
        if y_pred[j] > 0.5:
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
# 分别输出每一折对应的训练集和验证集的auc
if __name__ == '__main__':
    # 每一折保存的模型名称
    fold_model_dict = ['SwinTransformer_3C 第1折 0.8696 0.8188.pth',
                       'SwinTransformer_3C 第2折 0.8726 0.8516.pth',
                       'SwinTransformer_3C 第3折 0.8820 0.8391.pth',
                       'SwinTransformer_3C 第4折 0.8642 0.8438.pth',
                       'SwinTransformer_3C 第5折 0.9100 0.8516.pth',
                       'SwinTransformer_3C 第6折 0.8620 0.8750.pth',
                       'SwinTransformer_3C 第9折 0.8530 0.8520.pth',
                       'SwinTransformer_3C 第7折 0.8644 0.8406.pth',
                       'SwinTransformer_3C 第8折 0.8558 0.8219.pth',
                       'SwinTransformer_3C 第10折 0.8285 0.8300.pth'
                       ]
    test_dataset = getdataset("../dataset/lung.csv", "../dataset/roi/img_test0", "../dataset/fat_intrathoracic",
                              lung_transform['train'], mode_select=4, is_augment=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12)
    all_fold_test_auc = []
    all_fold_test_acc = []
    all_fold_test_sens = []
    all_fold_test_spec = []
    all_fold_test_ppv = []
    all_fold_test_npv = []

    # i 控制折数
    for i in range(len(fold_model_dict)):
        # 每一折创建一个新的模型，并加载该模型对应的参数
        model_dict_path = os.path.join("../runner/Model_Dict", fold_model_dict[i])
        model = swin_small_patch4_window7_224(num_classes=2)
        model.load_state_dict(torch.load(model_dict_path))
        model.cuda()
        model.eval()

        test_batch_true = []
        test_batch_pred = []

        # 计算训练集每一个batch的模型预测值，并保存
        for data in tqdm(test_dataloader):
            img, label = data
            img = img.to('cuda:0')
            pred = model(img).softmax(dim=1)[:, 1]
            pred = pred.cpu().detach().numpy()
            test_batch_pred.append(pred)
            test_batch_true.append(label)

        test_y_pred = np.concatenate(test_batch_pred)
        test_y_true = np.concatenate(test_batch_true)

        fpr, tpr, thresholds = roc_curve(test_y_true, test_y_pred)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        fold_roc_auc = auc(fpr, tpr)
        fold_acc, fold_sens, fold_spec, fold_ppv, fold_npv = cal_acc_sens_spec_ppv_npv(best_threshold, test_y_true,
                                                                                       test_y_pred)
        print(f"第{i + 1}折验证集AUC:{fold_roc_auc}, ACC:{fold_acc}, SENS:{fold_sens}, SPEC{fold_spec}:, PPV:{fold_ppv}, PPV:{fold_npv}")
        all_fold_test_auc.append(fold_roc_auc)
        all_fold_test_acc.append(fold_acc)
        all_fold_test_sens.append(fold_sens)
        all_fold_test_spec.append(fold_spec)
        all_fold_test_ppv.append(fold_ppv)
        all_fold_test_npv.append(fold_npv)

        # 计算最终的平均值，以及置信区间
    auc_mean, auc_ci = cal_mean_CI(all_fold_test_auc)
    acc_mean, acc_ci = cal_mean_CI(all_fold_test_acc)
    sens_mean, sens_ci = cal_mean_CI(all_fold_test_sens)
    spec_mean, spec_ci = cal_mean_CI(all_fold_test_spec)
    ppv_mean, ppv_ci = cal_mean_CI(all_fold_test_ppv)
    npv_mean, npv_ci = cal_mean_CI(all_fold_test_npv)

    print('AUC: {:.3f} [{:.3f}, {:.3f}]'.format(auc_mean, auc_ci[0], auc_ci[1]))
    print('ACC: {:.3f} [{:.3f}, {:.3f}]'.format(acc_mean, acc_ci[0], acc_ci[1]))
    print('SENS: {:.3f} [{:.3f}, {:.3f}]'.format(sens_mean, sens_ci[0], sens_ci[1]))
    print('SPEC: {:.3f} [{:.3f}, {:.3f}]'.format(spec_mean, spec_ci[0], spec_ci[1]))
    print('PPV: {:.3f} [{:.3f}, {:.3f}]'.format(ppv_mean, ppv_ci[0], ppv_ci[1]))
    print('NPV: {:.3f} [{:.3f}, {:.3f}]'.format(npv_mean, npv_ci[0], npv_ci[1]))