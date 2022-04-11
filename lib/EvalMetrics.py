#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2 as cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import scipy.optimize as optim
import copy
import random
import pickle
from IPython.display import Markdown, display
import seaborn as sns
import matplotlib.patches as patches
import sys
from LFR import *
from LFR import distances

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

def get_model_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, matrix, f1

def plot_model_performance(y_pred_s, y_pred_n, y_pred, y_true_s, y_true_n, y_true):
    accuracy_s, matrix_s, f1_s = get_model_performance(y_true_s, y_pred_s)

    display(Markdown('#### Sensitive data (Caucasians):'))
    print(f'Accuracy: {accuracy_s}')
    print(f'F1 score: {f1_s}')

    accuracy_n, matrix_n, f1_n = get_model_performance(y_true_n, y_pred_n)

    display(Markdown('#### Nonsensitive data (African-Americans):'))
    print(f'Accuracy: {accuracy_n}')
    print(f'F1 score: {f1_n}')

    accuracy, matrix, f1 = get_model_performance(y_true, y_pred)

    display(Markdown('#### All data:'))
    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {f1}')

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 3, 1)
    sns.heatmap(matrix_s, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix (sensitive data)')

    ax = fig.add_subplot(1, 3, 2)
    sns.heatmap(matrix_n, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix (nonsensitive data)')

    ax = fig.add_subplot(1, 3, 3)
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix (all data)')

def equal_opportunity_difference(y_test_s, y_test_n, pred_test_s, pred_test_n):
    tpr_s = 0
    for i in range(len(y_test_s)):
        if y_test_s[i] == 1 and pred_test_s[i] == 1:
            tpr_s += 1
    tpr_s = tpr_s/len(y_test_s)
    tpr_n = 0
    for i in range(len(y_test_n)):
        if y_test_n[i] == 1 and pred_test_n[i] == 1:
            tpr_n += 1
    tpr_n = tpr_n/len(y_test_n)

    equal_opportunity_difference = tpr_s - tpr_n

    return equal_opportunity_difference

def avg_abs_odds_difference(y_test_s, y_test_n, pred_test_s, pred_test_n):
    tpr_s = 0
    for i in range(len(y_test_s)):
        if y_test_s[i] == 1 and pred_test_s[i] == 1:
            tpr_s += 1
    tpr_s = tpr_s/len(y_test_s)
    tpr_n = 0
    for i in range(len(y_test_n)):
        if y_test_n[i] == 1 and pred_test_n[i] == 1:
            tpr_n += 1
    tpr_n = tpr_n/len(y_test_n)

    fpr_s = 0
    for i in range(len(y_test_s)):
        if y_test_s[i] == 0 and pred_test_s[i] == 1:
            fpr_s += 1
    fpr_s = fpr_s/len(y_test_s)
    fpr_n = 0
    for i in range(len(y_test_n)):
        if y_test_n[i] == 0 and pred_test_n[i] == 1:
            fpr_n += 1
    fpr_n = fpr_n/len(y_test_n)

    avg_abs_odds_diff = 0.5*(abs(fpr_s - fpr_n) + abs(tpr_s - tpr_n))

    return avg_abs_odds_diff

def fair_metrics(pred_test_s, pred_test_n, pred_test, y_test_s, y_test_n, y_test):

    cols = ['calibration', 'equal_opportunity_difference', 'average_abs_odds_difference',  'disparate_impact']
    obj_fairness = [[0,0,0,1]]

    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)

    acc_sen, acc_nsen, total_accuracy = calc_accuracy(pred_test_s, pred_test_n, y_test_s, y_test_n)

    calibration = acc_sen - acc_nsen

    equal_opp_diff = equal_opportunity_difference(y_test_s, y_test_n, pred_test_s, pred_test_n)

    avg_abs_odds_diff = avg_abs_odds_difference(y_test_s, y_test_n, pred_test_s, pred_test_n)

    disparate_impact = acc_sen/acc_nsen

    row = pd.DataFrame([[calibration, equal_opp_diff, avg_abs_odds_diff, disparate_impact]],
                           columns  = cols,
                           index = ['Race']
                          )

    fair_metrics = fair_metrics.append(row)
    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)

    return fair_metrics

def plot_fair_metrics(fair_metrics):
    fig, ax = plt.subplots(figsize=(20,4), ncols=5, nrows=1)

    plt.subplots_adjust(
        left    =  0.125,
        bottom  =  0.1,
        right   =  0.9,
        top     =  0.9,
        wspace  =  .5,
        hspace  =  1.1
    )

    y_title_margin = 1.2

    plt.suptitle("Fairness metrics", y = 1.09, fontsize=20)
    sns.set(style="dark")

    cols = fair_metrics.columns.values
    obj = fair_metrics.loc['objective']
    size_rect = [0.2,0.2,0.2,0.4]
    rect = [-0.1,-0.1,-0.1,0.8]
    bottom = [-1,-1,-1,0]
    top = [1,1,1,2]
    bound = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[0.8,1.2]]

    display(Markdown("### Check bias metrics :"))
    display(Markdown("A model can be considered bias if just one of these four metrics show that this model is biased."))
    for attr in fair_metrics.index[1:len(fair_metrics)].values:
        display(Markdown("#### For the %s attribute :"%attr))
        check = [bound[i][0] < fair_metrics.loc[attr][i] < bound[i][1] for i in range(0,4)]
        display(Markdown("With default thresholds, bias against unprivileged group detected in **%d** out of 4 metrics"%(5 - sum(check))))

    for i in range(0,4):
        plt.subplot(1, 4, i+1)
        ax = sns.barplot(x=fair_metrics.index[1:len(fair_metrics)], y=fair_metrics.iloc[1:len(fair_metrics)][cols[i]])

        for j in range(0,len(fair_metrics)-1):
            a, val = ax.patches[j], fair_metrics.iloc[j+1][cols[i]]
            marg = -0.2 if val < 0 else 0.1
            ax.text(a.get_x()+a.get_width()/4, a.get_y()+a.get_height()+marg, round(val, 3), fontsize=15,color='black')

        plt.ylim(bottom[i], top[i])
        plt.setp(ax.patches, linewidth=0)
        ax.add_patch(patches.Rectangle((-5,rect[i]), 10, size_rect[i], alpha=0.3, facecolor="green", linewidth=1, linestyle='solid'))
        plt.axhline(obj[i], color='black', alpha=0.3)
        plt.title(cols[i])
        ax.set_ylabel('')
        ax.set_xlabel('')
