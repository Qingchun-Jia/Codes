# encoding=utf-8
import numpy as np
import pandas as pd
import radiomics
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import SimpleITK as sitk
from Utils.data_transform import lung_transform
from Utils.get_dataset import getdataset
from Utils.get_ten_fold_dataset import get_ten_fold_dataloaders
from models.cla_models import swin_small_patch4_window7_224
feature_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
feature_extractor.enableAllFeatures()
feature_extractor.enableAllImageTypes()
feature_extractor.enableFeatureClassByName('shape2D')

mode_select = 5
train_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_train",
                           "../../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False)

test0_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_test0",
                           "../../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False)
test1_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_test1",
                           "../../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False)
test2_dataset = getdataset("../../dataset/lung.csv", "../../dataset/roi/img_test2",
                           "../../dataset/fat_intrathoracic",
                           lung_transform['train'], mode_select=mode_select, is_augment=False)

all_dataset = [train_dataset, test0_dataset, test1_dataset, test2_dataset]
all_csv_name = ['train_data_rad', 'val_data_rad', 'test0_data_rad', 'test1_data_rad', 'test2_data_rad']
i=0
for dataset in all_dataset:
    name_list = dataset.img_path_list
    label_list = dataset.label_list
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    feature_df = pd.DataFrame()
    for data in tqdm(data_loader):
        img, _ = data
        img = np.squeeze(img.numpy())
        mask = np.ones_like(img)
        img_sitk = sitk.GetImageFromArray(img)
        mask_sitk = sitk.GetImageFromArray(mask)
        radiomic_features = feature_extractor.execute(img_sitk, mask_sitk)
        # 将特征转换为DataFrame的行
        feature_row = pd.DataFrame([radiomic_features])
        # 将特征行添加到DataFrame中
        feature_df = feature_df.append(feature_row, ignore_index=True)
    print(i)
    id_list = [id[28:-4] for id in name_list]
    feature_df.insert(0, 'Image_ID', id_list)
    feature_df.insert(1, 'Label', label_list)
    feature_df.to_csv(f'radiomic_features_save/{all_csv_name[i]}.csv', index=False)
    i=i+1


