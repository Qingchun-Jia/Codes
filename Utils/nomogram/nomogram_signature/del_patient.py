import pandas as pd
#
del_list = [23419698, 24729121, 24707879, 24410049, 24654407,
            24893177, 24851593, 20779929, 24677904, 24871660, 24681509, 24737126]
for fold in range(1, 11):
    data = pd.read_csv(f'../nomogram_signature/fold_{fold}/test1_data_{fold}.csv')
    data = data[~data['Image_ID'].isin(del_list)]
    data.to_csv(f'../nomogram_signature/fold_{fold}/test1_data_{fold}.csv', index=False)