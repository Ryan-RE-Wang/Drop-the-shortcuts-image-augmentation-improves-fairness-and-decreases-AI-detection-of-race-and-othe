from sklearn.metrics import confusion_matrix, f1_score, roc_curve
import numpy as np

def get_threshes(y_test, preds):
    tprs = []
    fprs = []
    thresholds = []

    for i in range(14):
        fpr, tpr, _ = roc_curve(y_test[:, i], preds[:, i], drop_intermediate=False)
        
        tprs.append(tpr)
        fprs.append(fpr)
        thresholds.append(_)
        
    return tprs, fprs, thresholds

def get_tpr(y_test, preds, thresh):
    tn, fp, fn, tp = confusion_matrix(y_test, np.where(preds >= thresh, 1, 0)).ravel()
    
    return tp/(tp+fn)

def get_fpr(y_test, preds, thresh):
    tn, fp, fn, tp = confusion_matrix(y_test, np.where(preds >= thresh, 1, 0)).ravel()
    
    return fp/(fp+tn)

def cal_best_thresh(y_test, y_preds):
    best_thresh = [0]* 14
    tprs, fprs, threshes = get_threshes(y_test, y_preds)
        
    for j in range(14):
        best = -1
        for i in range(len(threshes[j])):
            score = f1_score(y_test[:, j], np.where(y_preds[:, j] > threshes[j][i], 1, 0), average='binary')
            if (score > best):
                best = score
                best_thresh[j] = threshes[j][i]
                
    return best_thresh