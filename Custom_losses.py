import tensorflow as tf
import numpy as np

group_type = {'race': [0, 1, 4], 'gender': [0, 1], 'age': [0, 1, 2, 3]}

def my_cdist(x1, x2):
    x1_norm = tf.reduce_sum(tf.math.pow(x1, 2), axis=-1, keepdims=True)
    x2_norm = tf.reduce_sum(tf.math.pow(x2, 2), axis=-1, keepdims=True)
    
    mm = -2*tf.linalg.matmul(x1, tf.transpose(x2, perm=[1, 0]))

    madd = tf.math.add(tf.transpose(x2_norm, perm=[1, 0]), mm)

    res = tf.math.add(madd, x1_norm)

    return tf.clip_by_value(res, clip_value_min=1e-30, clip_value_max=1e30)

def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                       1000]):
    D = my_cdist(x, y)
    
    K = tf.zeros_like(D)

    for g in gamma:
        
        K = tf.math.add(K, tf.math.exp(tf.math.multiply(D, -g)))

    return K/len(gamma)

def mmd(x, y):
    # https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
    Kxx = tf.reduce_mean(gaussian_kernel(x, x))
    Kyy = tf.reduce_mean(gaussian_kernel(y, y))
    Kxy = tf.reduce_mean(gaussian_kernel(x, y))
    return Kxx + Kyy - 2 * Kxy

def DistMatch_Loss(y_test, y_preds, demo, mode, group):        
    penalty = 0

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    bce_loss = bce(y_test, y_preds)
    
    dist_all_target = y_preds
        
    for target in group_type[group]:
        mask = demo==target
                
        dist_grp_target = y_preds[mask]
        
        if len(dist_grp_target) > 0:
            
            if (mode == 'MMD'):   
                penalty += mmd(dist_grp_target, dist_all_target)
            elif (mode == 'Mean'):
                penalty += tf.math.abs(tf.reduce_mean(dist_grp_target) -  tf.reduce_mean(dist_all_target))
            else:
                print('Wrong mode!')
                return

    return bce_loss, penalty



surrogate_fns = [
    lambda t,y: tpr_surrogate(t,y, threshold = 0.1, surrogate_fn = tf.math.sigmoid),
    lambda t,y: fpr_surrogate(t,y, threshold = 0.1, surrogate_fn = tf.math.sigmoid)
]

def tpr_surrogate(
    labels, outputs, threshold=0.1, surrogate_fn=None
):
    """
        The true positive rate (recall/sensitivity)
    """

    mask = labels == 1

    return tf.reduce_mean(surrogate_fn(outputs[mask] - threshold))

def fpr_surrogate(
    labels, outputs, threshold=0.1, surrogate_fn=None
):
    """
        The false positive rate (1-specificity)
    """
    mask = labels == 0

    return tf.reduce_mean(surrogate_fn(outputs[mask] - threshold))

group_type = {'race': [0, 1, 4], 'gender': [0, 1], 'age': [0, 1, 2, 3]}

def fairALM_loss(y_test, y_preds, demo, lag_mult, group, val):

    lag_increments = []
    penalty = 0
    eta = 1e-3

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    bce_loss = bce(y_test, y_preds)
        
    for i, target in enumerate(group_type[group]):
        mask = demo == target
        
        if tf.reduce_sum(y_test[mask]) == 0 or tf.reduce_sum(y_test[mask]) == len(y_test[mask]):
            for c, fn in enumerate(surrogate_fns):
                lag_increments.append(0)
            continue

        for c, fn in enumerate(surrogate_fns):
            lag_ind = i * len(surrogate_fns) + c
            grp_val = fn(y_test[mask], y_preds[mask])
            all_val = fn(y_test, y_preds)
            penalty += (lag_mult[lag_ind] + eta) * grp_val - (lag_mult[lag_ind] - eta) * all_val # conditional with marginal
            lag_increments.append(eta * (grp_val - all_val))
            
    if not (val):      
        lag_mult += lag_increments

    return bce_loss, penalty*100, lag_mult

def get_onehot(label, task):
    
    labels = []
    
    for i in label:
        if (task == 'race'):

            if (i == 0):
                labels.append([1, 0, 0])
            elif (i == 1):
                labels.append([0, 1, 0])
            else:
                labels.append([0, 0, 1])

        elif (task == 'age'):

            if (i == 0):
                labels.append([1, 0, 0, 0])
            elif (i == 1):
                labels.append([0, 1, 0, 0])
            elif (i == 2):
                labels.append([0, 0, 1, 0])
            else:
                labels.append([0, 0, 0, 1])

        elif (task == 'gender'):

            if (i == 0):
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            
    return np.array(labels)

def reciprocal_CCE_loss(y_test, y_preds):
    
    if (y_preds.shape[1] == 3):
        y_test = get_onehot(y_test, 'race')
    elif (y_preds.shape[1] == 2):
        y_test = get_onehot(y_test, 'gender')
    else:
        y_test = get_onehot(y_test, 'age')
        
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return 1 / (cce(y_test, y_preds) * 0.01)

def ERM_Loss(y_test, y_preds):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    bce_loss = bce(y_test, y_preds)
    
    return bce_loss