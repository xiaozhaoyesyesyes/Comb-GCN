#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle
import dataset
from model import BGCN, BGCN_Info
from utils import check_overfitting, early_stop, logger
from train import train
from metric import Recall, NDCG, Acc,f1_score,MRR
from config import CONFIG
from test import test
import loss
from itertools import product
import time
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def main():
    #  set env
    setproctitle.setproctitle(f"train{CONFIG['name']}")#setproctitle.setproctitle("进程别名")
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    device = torch.device('cuda')

    #  fix seed
    seed = 123
    random.seed(seed)#当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的，同时选择不同的参数生成的随机数也不一样
    os.environ['PYTHONHASHSEED'] = str(seed)#设置环境变量=环境变量值
    np.random.seed(seed)#生成随机数、起始位置在123
    torch.manual_seed(seed)#设置固定生成随机数的种子，使得每次运行该 .py 文件时生成的随机数相同cpu
    torch.cuda.manual_seed(seed)#设置固定生成随机数的种子，使得每次运行该 .py 文件时生成的随机数相同gpu
    torch.cuda.manual_seed_all(seed)#为所有的 GPU 设置种子用于生成随机数，以使得结果是确定的。
    torch.backends.cudnn.deterministic = True#flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    #  load data 载入数据
    # 五折交叉验证
    # kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    #
    # train_index = []
    # test_index = []
    #
    # for train_idx, test_idx in kf.split(samples[:, 2]):
    #     train_index.append(train_idx)
    #     test_index.append(test_idx)
    bundle_train_data, bundle_test_data, item_data, assist_data = \
            dataset.get_dataset(CONFIG['path'], CONFIG['dataset_name'], task=CONFIG['task'])

    train_loader = DataLoader(bundle_train_data, 1, True,
                              num_workers=8, pin_memory=True)#DataLoader(数据集、batch_size、shuffle=true、num_workers即加载数据使用几个进程，pin_menmory作用就是从一开始就把一部分内存给锁住（上图（右）），这样一来就减少了Host内部的开销，避免了CPU内存拷贝时间)
    test_loader = DataLoader(bundle_test_data, 1, False,
                             num_workers=8, pin_memory=True)

    # print("===============")
    # for step, (b_x, b_y) in enumerate(train_loader):
    #     if step > 0:
    #         break
    #     print("b_x.shape", b_y.shape)
    #     print("b_x.dtype", b_y.dtype)
    # print("===============")

    #  pretrain
    if 'pretrain' in CONFIG:
        pretrain = torch.load(CONFIG['pretrain'], map_location='cpu')
        print('load pretrain')

    #  graph
    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    #  metric
    metrics = [Recall(20), NDCG(20),Acc(20),f1_score(20),MRR(20),Recall(40), NDCG(40),Acc(40),f1_score(40),MRR(40), Recall(80), NDCG(80),Acc(80),f1_score(80),MRR(80)]
    TARGET = 'Recall@80'

    #  loss
    loss_func = loss.BPRLoss('mean')#BPR Loss是用得比较多的一种raking loss

    #  log
    log = logger.Logger(os.path.join(
        CONFIG['log'], CONFIG['dataset_name'], 
        f"{CONFIG['model']}_{CONFIG['task']}", ''), 'best', checkpoint_target=TARGET)#输出日志
    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))  # 把时间转化为字符串
    theta = 0.6

    for lr, decay, message_dropout, node_dropout \
            in product(CONFIG['lrs'], CONFIG['decays'], CONFIG['message_dropouts'], CONFIG['node_dropouts']):
        # decay为学习率衰减因子，减少训练过程的震荡
        visual_path =  os.path.join(CONFIG['visual'], 
                                    CONFIG['dataset_name'],  
                                    f"{CONFIG['model']}_{CONFIG['task']}", 
                                    f"{time_path}@{CONFIG['note']}", 
                                    f"lr{lr}_decay{decay}_medr{message_dropout}_nodr{node_dropout}")

        # model
        if CONFIG['model'] == 'BGCN':
            graph = [ub_graph, ui_graph, bi_graph]#三部图
            info = BGCN_Info(64, decay, message_dropout, node_dropout, 2)#64为embeddingsize,2为节点层
            model = BGCN(info, assist_data, graph, device, pretrain=None).to(device)


        assert model.__class__.__name__ == CONFIG['model']

        # op
        op = optim.Adam(model.parameters(), lr=lr)
        # env
        env = {'lr': lr,
               'op': str(op).split(' ')[0],   # Adam
               'dataset': CONFIG['dataset_name'],
               'model': CONFIG['model'], 
               'sample': CONFIG['sample'],
               }

        #  continue training
        if CONFIG['sample'] == 'hard' and 'conti_train' in CONFIG:
            model.load_state_dict(torch.load(CONFIG['conti_train']))
            # model.load_state_dict(torch.load('/home/heq/xinyi/workspace/log/NetEase/BGCN_tune/04-07-14-50-03-some_note/1_6e4967_Recall@80.pth'))

            print('load model and continue training')


        retry = CONFIG['retry']  # =1
        while retry >= 0:
            # log
            log.update_modelinfo(info, env, metrics)

            # try:
            # train & test
            early = CONFIG['early']   #early=50
            train_writer = SummaryWriter(log_dir=visual_path, comment='train')
            train_writer = SummaryWriter(comment='train')
            test_writer = SummaryWriter(log_dir=visual_path, comment='test')
            test_writer = SummaryWriter(comment='test')
            for epoch in range(CONFIG['epochs']):
                # train
                trainloss = train(model, epoch+1, train_loader, op, device, CONFIG, loss_func)
                train_writer.add_scalars('loss/single', {"loss": trainloss}, epoch)
               # train_writer.add_scalars('Accuracy', {"Accuracy": train_accuracy}, epoch)

                # test
                if epoch % CONFIG['test_interval'] == 0:
                    output_metrics = test(model, test_loader, device, CONFIG, metrics)

                    for metric in output_metrics:
                        test_writer.add_scalars('metric/all', {metric.get_title(): metric.metric}, epoch)
                        if metric==output_metrics[5]:
                            test_writer.add_scalars('metric/single', {metric.get_title(): metric.metric}, epoch)
                    # log
                    log.update_log(metrics, model)

                    # 0
                    if epoch > 10:
                        if check_overfitting(log.metrics_log, TARGET, 1, show=False):
                            break
                    # early stop
                    #early = early_stop(
                    #    log.metrics_log[TARGET], early, threshold=0)
                    #if early <= 0:
                    #    break
            train_writer.close()
            test_writer.close()

            log.close_log(TARGET)
            retry = -1
            # except RuntimeError:
            #     retry -= 1
    log.close()

# 定义训练精确度存入的文件
# df = pd.DataFrame(columns=['time', 'step', 'train Loss', 'training accuracy'])  # 列名
# df.to_csv("acc.csv", index=False)  # 路径可以根据需要更改
# print("acc have created!")
if __name__ == "__main__":
    main()
