#encoding=utf-8
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# 根据第八折的结果，构造Age,Sex,IPN,Adipose,IPN+Adipose,IPN+Adipose+Age+Sex的所有诺莫图预测结果，方便后续画图
# 先训练每一个特征对应的Logist模型
nomogram_data = pd.read_csv(f'../nomogram/nomogram_signature/fold_8/train_data_8.csv')
label = np.array(nomogram_data.iloc[:, 1]).astype(int)
IPN_signature = np.reshape(np.array(nomogram_data.iloc[:, 2]).astype(float), (-1, 1))
Adipose_signature = np.reshape(np.array(nomogram_data.iloc[:, 3]).astype(float), (-1, 1))
Sex_signature = np.reshape(np.array(nomogram_data.iloc[:, 4]).astype(float), (-1, 1))
Age_signature = np.reshape(np.array(nomogram_data.iloc[:, 5]).astype(float), (-1, 1))
Age_Sex_signature = np.concatenate((Sex_signature, Age_signature), axis=1)
Age_Sex_IPN_signature = np.concatenate((Sex_signature, Age_signature, IPN_signature), axis=1)
IPN_Adipose_Age_Sex_signature = np.concatenate((IPN_signature, Adipose_signature, Age_signature, Sex_signature), axis=1)

Age_Logist_regressor = LogisticRegression()
Age_Logist_regressor.fit(Age_signature, label)
joblib.dump(Age_Logist_regressor, "./signature_8_pkl/Logist_regressor_Age.pkl")


Sex_Logist_regressor = LogisticRegression()
Sex_Logist_regressor.fit(Sex_signature, label)
joblib.dump(Sex_Logist_regressor, "./signature_8_pkl/Logist_regressor_Sex.pkl")

IPN_Logist_regressor = LogisticRegression()
IPN_Logist_regressor.fit(IPN_signature, label)
joblib.dump(IPN_Logist_regressor, "./signature_8_pkl/Logist_regressor_IPN.pkl")

Adi_Logist_regressor = LogisticRegression()
Adi_Logist_regressor.fit(Adipose_signature, label)
joblib.dump(Adi_Logist_regressor, "./signature_8_pkl/Logist_regressor_Adi.pkl")
#
Age_Sex_regressor = LogisticRegression()
Age_Sex_regressor.fit(Age_Sex_signature, label)
joblib.dump(Age_Sex_regressor, "./signature_8_pkl/Logist_regressor_Age_Sex.pkl")

Age_Sex_IPN_regressor = LogisticRegression()
Age_Sex_IPN_regressor.fit(Age_Sex_IPN_signature, label)
joblib.dump(Age_Sex_IPN_regressor, './signature_8_pkl/Logist_regressor_Age_Sex_IPN.pkl')


all_logist = LogisticRegression()
all_logist.fit(IPN_Adipose_Age_Sex_signature, label)
y_pred = all_logist.predict(IPN_Adipose_Age_Sex_signature)
joblib.dump(all_logist, './signature_8_pkl/Logist_regressor_all.pkl')


all_csv_name = ['train_data', 'val_data', 'test0_data', 'test1_data', 'test2_data']
for csv_name in all_csv_name:
    nomogram_data = pd.read_csv(f'../nomogram/nomogram_signature/fold_8/{csv_name}_8.csv')
    img_id = np.array(nomogram_data.iloc[:, 0]).astype(int)
    label = np.array(nomogram_data.iloc[:, 1]).astype(int)
    IPN_signature = np.reshape(np.array(nomogram_data.iloc[:, 2]).astype(float), (-1, 1))
    Adipose_signature = np.reshape(np.array(nomogram_data.iloc[:, 3]).astype(float), (-1, 1))
    Sex_signature = np.reshape(np.array(nomogram_data.iloc[:, 4]).astype(float), (-1, 1))
    Age_signature = np.reshape(np.array(nomogram_data.iloc[:, 5]).astype(float), (-1, 1))
    Age_Sex_signature = np.concatenate((Sex_signature, Age_signature), axis=1)
    Age_Sex_IPN_signature = np.concatenate((Sex_signature, Age_signature, IPN_signature), axis=1)
    IPN_Adipose_signature = np.concatenate((IPN_signature, Adipose_signature), axis=1)
    IPN_Adipose_Age_Sex_signature = np.concatenate((IPN_signature, Adipose_signature, Age_signature, Sex_signature),
                                                   axis=1)
    Age_Logistic_regressor = joblib.load("./signature_8_pkl/Logist_regressor_Age.pkl")

    Sex_Logistic_regressor = joblib.load("./signature_8_pkl/Logist_regressor_Sex.pkl")

    IPN_Logistic_regressor = joblib.load("./signature_8_pkl/Logist_regressor_IPN.pkl")

    Adi_Logistic_regressor = joblib.load("./signature_8_pkl/Logist_regressor_Adi.pkl")

    Age_Sex_regressor = joblib.load('./signature_8_pkl/Logist_regressor_Age_Sex.pkl')

    Age_Sex_IPN_Logistic_regressor = joblib.load('./signature_8_pkl/Logist_regressor_Age_Sex_IPN.pkl')

    all_Logistic_regressor = joblib.load('./signature_8_pkl/Logist_regressor_all.pkl')

    Age_pred = Age_Logistic_regressor.predict_proba(Age_signature)[:, 1]
    Sex_pred = Sex_Logistic_regressor.predict_proba(Sex_signature)[:, 1]
    IPN_pred = IPN_Logistic_regressor.predict_proba(IPN_signature)[:, 1]
    Adi_pred = Adi_Logistic_regressor.predict_proba(Adipose_signature)[:, 1]
    Age_Sex_pred = Age_Sex_regressor.predict_proba(Age_Sex_signature)[:, 1]
    Age_Sex_IPN_pred = Age_Sex_IPN_Logistic_regressor.predict_proba(Age_Sex_IPN_signature)[:, 1]
    all_pred = all_Logistic_regressor.predict_proba(IPN_Adipose_Age_Sex_signature)[:, 1]
    print(roc_auc_score(label, Age_pred), roc_auc_score(label, Sex_pred), roc_auc_score(label, IPN_pred)
          ,roc_auc_score(label, Adi_pred), roc_auc_score(label, Age_Sex_pred),roc_auc_score(label, Age_Sex_IPN_pred), roc_auc_score(label, all_pred))
    final_data = {'Image_ID': img_id, 'Label': label, 'Age_pred':Age_pred, 'Sex_pred':Sex_pred, 'IPN_pred':IPN_pred,
                  'Adi_pred':Adi_pred, 'Age_Sex_pred':Age_Sex_pred, 'Age_Sex_IPN_pred':Age_Sex_IPN_pred, 'all_pred':all_pred}
    df = pd.DataFrame(final_data)
    df.to_csv(f'signature_8_csv/{csv_name}.csv', index=False)
