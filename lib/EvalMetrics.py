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

# this function defines the threshold for y_n_hat to be 0 or 1
def predic_category(y):
    for i in range(len(y)):
        if y[i] >= 0.5:
            y[i] = 1
        else:
            y[i] = 0
    return y

# this function calculate y_n_hat by using the best parameters
def predict(params, data_sensitive, data_nonsensitive, k=10):
    
    Ns, P = data_sensitive.shape
    Nns, _ = data_nonsensitive.shape
    
    # form parameters in new forms
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
    
    # make predictions for sensitive data
    yhat_sensitive = np.zeros(Ns)
    for i in range(Ns):
        for j in range(k):
            yhat_sensitive[i] += M_nk_sensitive[i, j] * w[j]
        yhat_sensitive[i] = 1e-6 if yhat_sensitive[i] <= 0 else yhat_sensitive[i]
        yhat_sensitive[i] = 0.999 if yhat_sensitive[i] >= 1 else yhat_sensitive[i]
        
    # make predictions for nonsensitive data
    yhat_nonsensitive = np.zeros(Nns)
    for i in range(Nns):
        for j in range(k):
            yhat_nonsensitive[i] += M_nk_nonsensitive[i, j] * w[j]
        yhat_nonsensitive[i] = 1e-6 if yhat_nonsensitive[i] <= 0 else yhat_nonsensitive[i]
        yhat_nonsensitive[i] = 0.999 if yhat_nonsensitive[i] >= 1 else yhat_nonsensitive[i]
        
    final_y_s = predic_category(yhat_sensitive)
    final_y_n = predic_category(yhat_nonsensitive)
    
    return final_y_s, final_y_n

def calc_accuracy(y_sen, y_nsen, y_sen_label, y_nsen_label):
    y_sen_df = pd.DataFrame(y_sen)
    y_nsen_df = pd.DataFrame(y_nsen)
    y_label = pd.DataFrame(y_sen_label).append(pd.DataFrame(y_nsen_label))
    y_df = y_sen_df.append(y_nsen_df)
    
    acc_sen = accuracy_score(y_sen_label, y_sen_df)
    acc_nsen = accuracy_score(y_nsen_label, y_nsen_df)
    total_accuracy = accuracy_score(y_label, y_df)
    
    return acc_sen, acc_nsen, total_accuracy

def calc_calibration(acc_sen, acc_nsen):
    return abs(acc_sen - acc_nsen)

