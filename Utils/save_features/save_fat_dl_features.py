# encoding=utf-8
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset
from Utils.get_ten_fold_dataset import get_ten_fold_dataloaders
from models.cla_models import swin_small_patch4_window7_224

mode_select = 5

train_val_dataloader = get_ten_fold_dataloaders(csv_path="../../dataset/lung.csv",
                                                node_path="../../dataset/roi/img_train",
                                                fat_path="../../dataset/fat_intrathoracic",
                                                transform_mode='original',
                                                mode_select=mode_select)

fold_model_dict = ['SwinTransformer_fat 第1折 0.9492 0.8578.pth',
                   'SwinTransformer_fat 第2折 0.9518 0.9812.pth',
                   'SwinTransformer_fat 第3折 0.9471 0.9906.pth',
                   'SwinTransformer_fat 第4折 0.9368 0.9734.pth',
                   'SwinTransformer_fat 第5折 0.9404 0.9891.pth',
                   'SwinTransformer_fat 第6折 0.9438 0.9859.pth',
                   'SwinTransformer_fat 第7折 0.9326 0.9922.pth',
                   'SwinTransformer_fat 第8折 0.9284 0.9828.pth',
                   'SwinTransformer_fat 第9折 0.9606 0.9041.pth',
                   'SwinTransformer_fat 第10折 0.9503 0.9517.pth'
                   ]

# 第n折
for fold, (train_loader, val_loader) in enumerate(train_val_dataloader):
    if True:
        test0_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_test0",
                                   "../../dataset/fat_intrathoracic",
                                   lung_transform['original'], mode_select=mode_select, is_augment=False)
        test1_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_test1",
                                   "../../dataset/fat_intrathoracic",
                                   lung_transform['original'], mode_select=mode_select, is_augment=False)
        test2_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_test2",
                                   "../../dataset/fat_intrathoracic",
                                   lung_transform['original'], mode_select=mode_select, is_augment=False)
        test0_name_list = test0_dataset.img_path_list
        test1_name_list = test1_dataset.img_path_list
        test2_name_list = test2_dataset.img_path_list
        test0_dataloader = torch.utils.data.DataLoader(test0_dataset, batch_size=1)
        test1_dataloader = torch.utils.data.DataLoader(test1_dataset, batch_size=1)
        test2_dataloader = torch.utils.data.DataLoader(test2_dataset, batch_size=1)

        model = swin_small_patch4_window7_224(num_classes=2)
        model.load_state_dict(torch.load(f"../../runner/Model_Dict/{fold_model_dict[fold]}"))
        model.eval()
        model.cuda()
        train_name_list = []
        val_name_list = []
        augment_train_name_list = train_loader.dataset.datasets[0].img_path_list + train_loader.dataset.datasets[
            1].img_path_list
        augment_val_name_list = val_loader.dataset.datasets[0].img_path_list + val_loader.dataset.datasets[
            1].img_path_list
        # 用来存储图像id号
        for sam in train_loader.sampler:
            train_name_list.append(augment_train_name_list[sam])
        for sam2 in val_loader.sampler:
            val_name_list.append(augment_val_name_list[sam2])

        all_dataloader = [train_loader, val_loader, test0_dataloader, test1_dataloader, test2_dataloader]
        all_data_name = [train_name_list, val_name_list, test0_name_list, test1_name_list, test2_name_list]
        all_csv_name = ['train_data', 'val_data', 'test0_data', 'test1_data', 'test2_data']

        for i in range(len(all_dataloader)):
            # 标签，深度学习特征，预测值
            label_list = []
            feature_list = []
            pred_list = []
            for data in tqdm(all_dataloader[i]):
                imgs, labels = data
                imgs = imgs.cuda()
                with torch.no_grad():
                    features = model.forward_features(imgs)
                    y_pred = model(imgs).softmax(dim=1)[:, 1]
                features = features.cpu()
                feature_list.append(features)
                label_list.append(labels.cpu().numpy())
                pred_list.append(np.array(y_pred.cpu()))

            name_list = all_data_name[i]
            for j in range(len(name_list)):
                name_list[j] = name_list[j][28:-4]

            for k in range(len(feature_list)):
                feature_list[k] = feature_list[k].numpy()
                feature_list[k] = np.squeeze(feature_list[k], 0)
                feature_list[k] = feature_list[k].tolist()
            id = [str(sublist) for sublist in name_list]
            label = [int(sublist) for sublist in label_list]
            feature = [tuple(sublist) for sublist in feature_list]
            final_data = {'Image_ID': id, 'Label': label}
            for l in range(768):
                final_data[f'Feature_{l + 1}'] = [feat[l] for feat in feature_list]
            df = pd.DataFrame(final_data)
            df.to_csv(f'fat_fold_data/fold_{fold + 1}/{all_csv_name[i]}_{fold + 1}.csv', index=False)

            pred_list = np.concatenate(pred_list)
            label_list = np.concatenate(label_list)
            print(all_csv_name[i])
            print(roc_auc_score(label_list, pred_list))
