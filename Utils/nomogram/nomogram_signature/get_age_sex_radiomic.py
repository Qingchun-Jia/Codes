import pandas as pd

age_sex_data = pd.read_csv('age_sex.csv', encoding='GBK', usecols=[0, 2, 3])
all_csv_name = ['train_data', 'val_data', 'test0_data', 'test1_data', 'test2_data']

for fold in range(1, 11):
    for csv_name in all_csv_name:
        data = pd.read_csv(f'../nomogram_signature/fold_{fold}/{csv_name}_{fold}.csv')
        # 合并 data 和 age_sex_data 基于共同的 id 列
        merged_data = pd.merge(data, age_sex_data, on='Image_ID', how='left')

        # 将合并后的数据存储到新的文件
        merged_data.to_csv(f'../nomogram_signature/fold_{fold}/{csv_name}_{fold}.csv', index=False)
