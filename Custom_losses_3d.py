import torch
import numpy as np
import torch.nn as nn

group_type = {'race': [0, 1], 'gender': [0, 1], 'age': [0, 1]}

def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            
    res = torch.addmm(x2_norm.unsqueeze(1),
                      x1.unsqueeze(1),
                      x2.unsqueeze(1).transpose(-1, 0), alpha=-2).add_(x1_norm.unsqueeze(1))
        
    return res.clamp_min_(1e-30)

def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                       1000]):
    D = my_cdist(x, y)
    K = torch.zeros_like(D)

    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))

    return K/len(gamma)

def mmd(x, y):
    # https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
    Kxx = gaussian_kernel(x, x).mean()
    Kyy = gaussian_kernel(y, y).mean()
    Kxy = gaussian_kernel(x, y).mean()
    return Kxx + Kyy - 2 * Kxy

def DistMatch_Loss(y_test, y_preds, demo, mode, group): 
    
    y_test = y_test.cpu()
    y_preds = y_preds.cpu()
    
    penalty = 0

    criterion_raw = nn.BCEWithLogitsLoss(reduction='none')
    
    bce_loss = criterion_raw(y_test, y_preds)
    
    dist_all_target = y_preds
            
    for target in group_type[group]:
        mask = demo==target
                
        dist_grp_target = y_preds[mask]
                
        if len(dist_grp_target) > 0:
            
            if (mode == 'MMD'):   
                penalty += mmd(dist_grp_target, dist_all_target)
            elif (mode == 'Mean'):
                penalty += torch.abs(torch.mean(dist_grp_target) -  torch.mean(dist_all_target))
            else:
                print('Wrong mode!')
                return

    return bce_loss, penalty

surrogate_fns = [
    lambda t,y: tpr_surrogate(t,y, threshold = 0.1, surrogate_fn = torch.sigmoid),
    lambda t,y: fpr_surrogate(t,y, threshold = 0.1, surrogate_fn = torch.sigmoid)
]

def tpr_surrogate(
    labels, outputs, threshold=0.1, surrogate_fn=None
):
    """
        The true positive rate (recall/sensitivity)
    """

    mask = labels == 1

    return torch.mean(surrogate_fn(outputs[mask] - threshold))

def fpr_surrogate(
    labels, outputs, threshold=0.1, surrogate_fn=None
):
    """
        The false positive rate (1-specificity)
    """
    mask = labels == 0

    return torch.mean(surrogate_fn(outputs[mask] - threshold))

group_type = {'race': [0, 1, 4], 'gender': [0, 1], 'age': [0, 1, 2, 3]}

def fairALM_loss(y_test, y_preds, demo, lag_mult, group, val):

    lag_increments = []
    penalty = 0
    eta = 1e-3
    
    y_test = y_test.cpu()
    y_preds = y_preds.cpu()

    criterion_raw = nn.BCEWithLogitsLoss(reduction='none')
    
    bce_loss = criterion_raw(y_test, y_preds)
        
    for i, target in enumerate(group_type[group]):
        mask = demo == target
        
        if torch.sum(y_test[mask]) == 0 or torch.sum(y_test[mask]) == len(y_test[mask]):
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
        lag_mult += torch.tensor(lag_increments)

    return bce_loss, penalty*100, lag_mult


def reciprocal_BCE_loss(y_test, y_preds):
    
    y_test = y_test.cpu()
    y_preds = y_preds.cpu()
        
    criterion_raw = nn.BCEWithLogitsLoss(reduction='none')
    return 1 / (criterion_raw(y_test, y_preds) * 0.01)

def ERM_Loss(y_test, y_preds):
    
    y_test = y_test.cpu()
    y_preds = y_preds.cpu()
    
    criterion_raw = nn.BCEWithLogitsLoss(reduction='none')
    
    bce_loss = criterion_raw(y_test, y_preds)
    
    return bce_loss