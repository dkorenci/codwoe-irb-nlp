"""
This script contains different implementations for losses
"""

# CKA implementation
# web: https://github.com/jayroxis/CKA-similarity/blob/main/CKA.py
# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

import math
import torch
import numpy as np
import torch.nn as nn


class CkaLoss(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T        
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA_loss(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA_loss(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    
class CkaLossTorch(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA_loss(self, X, Y):
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return 1 - hsic / (var1 * var2)

    def kernel_CKA_loss(self, X, Y, sigma=None):
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return 1 - hsic / (var1 * var2)

    
def mse_loss(pred, gt, *args, **kwargs):
    loss=nn.MSELoss()
    return loss(pred, gt)


def cos_loss(pred, gt, *args, **kwargs):
    device = device='cuda:{}'.format(pred.get_device())
    loss = nn.CosineEmbeddingLoss(
        margin=0.0, size_average=None, reduce=None, reduction='mean')
    return loss(pred, gt, torch.ones(gt.shape[0],device=device))


def cka_loss(pred, gt, **kwargs):
    device='cuda:{}'.format(pred.get_device())
    loss = CkaLossTorch(device).linear_CKA_loss
    return loss(pred, gt)


def xen_loss(pred, gt, ignore_index=None, **kwargs):
    if ignore_index==None:
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss(pred, gt)


def xen_smooth_loss(pred, gt, ignore_index=None, label_smoothing=0, **kwargs):
    if label_smoothing == 0:
        if ignore_index==None:
            loss = nn.CrossEntropyLoss()
        else:
            loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        if ignore_index==None:
            loss = nn.LabelSmoothingCrossEntropy(epsilon=label_smoothing)
        else:
            loss = nn.LabelSmoothingCrossEntropy(ignore_index=ignore_index, epsilon=label_smoothing)
    return loss(pred, gt)


def acc_gloss(pred, gt, ignore_index=None, **kwargs):
    # print('\nPRED', pred.shape)
    # print('GT', gt.shape)
    # print('PRED_ARGMAX', (pred.argmax(-1)).shape)
    # print('INDEX', ignore_index)
    
    tokens = gt != ignore_index
    
    # print('TOKENS', tokens.shape)
    # print('2222', ((pred.argmax(-1) == gt) & tokens).float().sum())
    # print('3333', tokens.sum())
    acc = (
        ((pred.argmax(-1) == gt) & tokens).float().sum() / tokens.sum()
    )
    return acc


LOSSES = {
    'mse': mse_loss,
    'cos': cos_loss,
    'cka': cka_loss,
    'xen': xen_loss,
    'xens': xen_smooth_loss,
}