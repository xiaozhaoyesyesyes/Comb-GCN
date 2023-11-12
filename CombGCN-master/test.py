#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt

def test(model, loader, device, CONFIG, metrics):
    '''
    test for dot-based model
    '''
    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        rs = model.propagate() 
        for users, ground_truth_u_b, train_mask_u_b in loader:
            pred_b = model.evaluate(rs, users.to(device))
            pred_b -= 1e8*train_mask_u_b.to(device)
            # confusion_matrixs = confusion_matrix(ground_truth_u_b.to(device), pred_b)
            # print("Confusion Matrix: \n", confusion_matrixs)
            # disp = plot_precision_recall_curve(classifier, X_test, y_test)
            # disp.ax_.set_title('P-R Example')
            # precision, recall, _thresholds = precision_recall_curve(ground_truth_u_b.cpu().to(device), pred_b.cpu())
            # auc = auc(recall, precision)
            # print("AUC: ", auc)

            for metric in metrics:
                metric(pred_b, ground_truth_u_b.to(device))
    print('Test: time={:d}s'.format(int(time()-start)))
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    return metrics

