import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc, log_loss, roc_auc_score
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def delong_test(y_true, model1, model2):
    # Convert to numpy array
    y_true = np.array(y_true)
    y_pred = np.array(model1)
    y_pred_2 = np.array(model2)

    # Calculate AUC values for both classifiers
    auc1 = roc_auc_score(y_true, y_pred)
    auc2 = roc_auc_score(y_true, y_pred_2)
    # Calculate difference in AUC values
    delta = auc1 - auc2

    # Calculate variance of the difference
    n1 = y_pred.shape[0]
    n2 = y_pred_2.shape[0]
    var = (auc1 * (1 - auc1) + auc2 * (1 - auc2)) / (n1 + n2 - 2)

    # Calculate z-score
    z_score = delta / np.sqrt(var)

    # Calculate p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return p_value


if __name__ == '__main__':
    all_csv_name = ['train', 'val', 'test0', 'test1', 'test2']
    for i in range(len(all_csv_name)):
        data = pd.read_csv(f'../Utils/nomogram/signature_8_csv/{all_csv_name[i]}_data.csv')
        y_true = np.array(data.iloc[:, 1]).astype(int)
        # age
        y_pred1 = np.array(data.iloc[:, 2]).astype(float)
        # sex
        y_pred2 = np.array(data.iloc[:, 3]).astype(float)
        # IPN
        y_pred3 = np.array(data.iloc[:, 4]).astype(float)
        # Adipose
        y_pred4 = np.array(data.iloc[:, 5]).astype(float)
        # age_sex
        y_pred5 = np.array(data.iloc[:, 6]).astype(float)
        # age_sex_ipn
        y_pred6 = np.array(data.iloc[:, 7]).astype(float)
        # nomogram
        y_pred7 = np.array(data.iloc[:, 8]).astype(float)


        age_nomogram_p = delong_test(y_true, y_pred1, y_pred7)
        sex_nomogram_p = delong_test(y_true, y_pred2, y_pred7)
        IPN_nomogram_p = delong_test(y_true, y_pred3, y_pred7)
        Adi_nomogram_p = delong_test(y_true, y_pred4, y_pred7)
        Age_Sex_nomogram_p = delong_test(y_true, y_pred5, y_pred7)
        Age_Sex_ipn_nomogram_p = delong_test(y_true, y_pred6, y_pred7)
        print('age:', age_nomogram_p, 'sex:', sex_nomogram_p, 'IPN', IPN_nomogram_p,
              "Adi:", Adi_nomogram_p, "Age_Sex:", Age_Sex_nomogram_p, "Age_sex_ipn:", Age_Sex_ipn_nomogram_p)
