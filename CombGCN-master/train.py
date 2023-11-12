#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
from metric import Recall, NDCG, Acc,f1_score
import os
from datetime import datetime
import pandas as pd

def train(model, epoch, loader, optim, device, CONFIG, loss_func):


    metrics = [Recall(20), NDCG(20), Acc(20),f1_score(20),Recall(40), NDCG(40), Acc(40),f1_score(40),Recall(80), NDCG(80), Acc(80),f1_score(80)]
    TARGET = 'Recall@80'

    log_interval = CONFIG['log_interval']
    print(log_interval)

    model.train()
    start = time()
    for i, (users_b, bundles) in enumerate(loader):

        # users_b, bundles = data
        # with torch.no_grad():
        modelout = model(users_b.to(device), bundles.to(device))
        loss = loss_func(modelout, batch_size=loader.batch_size)#一个batch里面有60个sample
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % log_interval == 0:
            print('U-B Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * loader.batch_size, len(loader.dataset),
                100. * (i+1) / len(loader), loss))
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time()-start)))
    return loss

