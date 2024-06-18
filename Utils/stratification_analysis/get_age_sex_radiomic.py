import pandas as pd
#
# age_sex_data = pd.read_csv('age_sex.csv', encoding='GBK', usecols=[0, 2, 3])
#
# data = pd.read_csv(f'../stratification_analysis/stratification_train.csv')
# # 合并 data 和 age_sex_data 基于共同的 id 列
# merged_data = pd.merge(data, age_sex_data, on='Image_ID', how='left')
#
# # 将合并后的数据存储到新的文件
# merged_data.to_csv(f'../stratification_analysis/stratification_train.csv', index=False)

thick_sys_data = pd.read_csv('BMI.csv', usecols=[0, 5])
data = pd.read_csv(f'../stratification_analysis/stratification_train.csv')
merged_data = pd.merge(data, thick_sys_data, on='Image_ID', how='left')
merged_data.to_csv(f'../stratification_analysis/stratification_train.csv', index=False)
