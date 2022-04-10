#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2 as cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import scipy.optimize as optim
import copy
import random
import pickle
import sys

# this function returns the distance matrix
def distances(X, v, alpha, N, P, k):
    dists = np.zeros((N, k))
    for i in range(N):
        for p in range(P):
            for j in range(k):    
                dists[i, j] += (X[i, p] - v[j, p]) * (X[i, p] - v[j, p]) * alpha[p]
    return dists

# this function returns the M_nk
def M_nk(dists, N, k):
    M_nk = np.zeros((N, k))
    exp = np.zeros((N, k))
    denom = np.zeros(N)
    for i in range(N):
        for j in range(k):
            exp[i, j] = np.exp(-1 * dists[i, j])
            denom[i] += exp[i, j]
        for j in range(k):
            if denom[i]:
                M_nk[i, j] = exp[i, j] / denom[i]
            else:
                M_nk[i, j] = exp[i, j] / 1e-6
    return M_nk
 
# this function returns the M_k array
def M_k(M_nk, N, k):
    M_k = np.zeros(k)
    for j in range(k):
        for i in range(N):
            M_k[j] += M_nk[i, j]
        M_k[j] /= N
    return M_k

# this function reconstructs of X to x_n_hat and L_x
def x_n_hat(X, M_nk, v, N, P, k):
    x_n_hat = np.zeros((N, P))
    L_x = 0.0
    for i in range(N):
        for p in range(P):
            for j in range(k):
                x_n_hat[i, p] += M_nk[i, j] * v[j, p]
            L_x += (X[i, p] - x_n_hat[i, p]) * (X[i, p] - x_n_hat[i, p])
    return x_n_hat, L_x

# this function returns a list of prediction
def yhat(M_nk, y, w, N, k):
    yhat = np.zeros(N)
    L_y = 0.0
    for i in range(N):
        for j in range(k):
            yhat[i] += M_nk[i, j] * w[j]
        yhat[i] = 1e-6 if yhat[i] <= 0 else yhat[i]
        yhat[i] = 0.999 if yhat[i] >= 1 else yhat[i]
        L_y += -1 * y[i] * np.log(yhat[i]) - (1.0 - y[i]) * np.log(1.0 - yhat[i])
    return yhat, L_y


# this function returns the objective function we want to minimize
def LFR_objective(params, data_sensitive, data_nonsensitive, y_sensitive, 
        y_nonsensitive,  k=10, A_x = 1e-4, A_y = 0.1, A_z = 1000):
    LFR_objective.iters += 1 
    Ns, P = data_sensitive.shape
    Nns, _ = data_nonsensitive.shape
    
    alpha0 = params[:P]
    alpha1 = params[P : 2 * P]
    w = params[2 * P : (2 * P) + k]
    v = np.matrix(params[(2 * P) + k:]).reshape((k, P))
        
    dists_sensitive = distances(data_sensitive, v, alpha0, Ns, P, k)
    dists_nonsensitive = distances(data_nonsensitive, v, alpha1, Nns, P, k)

    M_nk_sensitive = M_nk(dists_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk(dists_nonsensitive, Nns, k)
    
    M_k_sensitive = M_k(M_nk_sensitive, Ns, k)
    M_k_nonsensitive = M_k(M_nk_nonsensitive, Nns, k)
    
    L_z = 0.0
    for j in range(k):
        L_z += abs(M_k_sensitive[j] - M_k_nonsensitive[j])

    x_n_hat_sensitive, L_x_sen = x_n_hat(data_sensitive, M_nk_sensitive, v, Ns, P, k)
    x_n_hat_nonsensitive, L_x_nsen = x_n_hat(data_nonsensitive, M_nk_nonsensitive, v, Nns, P, k)
    L_x = L_x_sen + L_x_nsen

    yhat_sensitive, L_y_sen = yhat(M_nk_sensitive, y_sensitive, w, Ns, k)
    yhat_nonsensitive, L_y_nsen = yhat(M_nk_nonsensitive, y_nonsensitive, w, Nns, k)
    L_y = L_y_sen + L_y_nsen

    objective = A_x * L_x + A_y * L_y + A_z * L_z

    return objective

LFR_objective.iters = 0

def LFR(X_train_s, X_train_n, y_train_s, y_train_n, K=10, A_x = 1e-4, A_y = 0.1, A_z = 1000, iter = 100):
    rez = np.random.uniform(size=data.shape[1] * 2 + K + data.shape[1] * K)
    bnd = []
    for i, k2 in enumerate(rez):
        if i < data.shape[1] * 2 or i >= data.shape[1] * 2 + K:
            bnd.append((None, None))
        else:
            bnd.append((0, 1))
    
    # minimize the metric by parameters alpha, w and v
    para, min_L, d = optim.fmin_l_bfgs_b(LFR_objective, x0=rez, epsilon=1e-5, 
                                         args=(X_train_s, X_train_n, y_train_s, y_train_n, K, A_z, A_x, A_y), 
                                         bounds = bnd, approx_grad=True, 
                                         maxfun=iter, maxiter=iter)
    
    return para

