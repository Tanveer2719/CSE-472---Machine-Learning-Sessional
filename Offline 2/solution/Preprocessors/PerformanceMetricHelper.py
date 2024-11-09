import numpy as np
from sklearn.metrics import confusion_matrix,f1_score, roc_auc_score, average_precision_score


def custom_accuracy(y_true, y_pred):
    cor_pred = np.sum(y_true == y_pred)
    tot_samp = len(y_pred)

    return cor_pred/tot_samp
    

def custom_precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))

    return tp/(tp+fp) if (tp+fp)>0 else 0

def custom_sensitivity(y_true, y_pred):
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # tp = np.sum((y_true==1) & (y_pred==1))
    # fn = np.sum((y_true==1) & (y_pred==0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp/(tp+fn) if (tp+fn)>0 else 0

def custom_specificity(y_true, y_pred):


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))

    return tn/(tn+fp) if (tn+fp)>0 else 0


def f1_score_(y_true, y_pred):
    return f1_score(y_true, y_pred)

def auroc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def aupr(y_true, y_pred):
    return average_precision_score(y_true, y_pred)



