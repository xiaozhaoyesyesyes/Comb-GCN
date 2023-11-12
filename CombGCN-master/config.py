#!/usr/bin/env python3
# -*- coding: utf-8 -*-


CONFIG = {
    'name': '@changjianxin',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0",
    'note': 'some_note',
    'model': 'BGCN',
    'dataset_name': 'NetEase',
    'task': 'tune',
    'eval_task': 'test',

    ## search hyperparameters
    #  'lrs': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    #  'message_dropouts': [0, 0.1, 0.3, 0.5],
    #  'node_dropouts': [0, 0.1, 0.3, 0.5],
    #  'decays': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],

    ## optimal hyperparameters 
    'lrs': [3e-4],
    #'message_dropouts': [0.3],
    'message_dropouts': [0.1],
    'node_dropouts': [0],
    'decays': [1e-7],

    ## hard negative sample and further train
     #'sample': 'simple',
    'sample': 'hard',
    'hard_window': [0.7, 1.0], # top 30%
    'hard_prob': [0.4, 0.4], # probability 0.8
    #'hard_prob': [0.8, 0.8], # probability 0.8
    #'conti_train': 'model_file_from_simple_sample.pth',
    # 'conti_train': 'D:\\remote-server\\BGCN-master\\log\\NetEase\\BGCN_tune\\04-07-14-50-03-some_note\\model.csv',
    'conti_train': '/home/heq/xinyi/workspace/log/NetEase/BGCN_tune/05-27-18-28-54-some_note/1_2918af_Recall@80.pth',

    ## other settings
    'epochs': 100,
    'early': 50,#Early Stop是训练复杂机器学习模型以避免其过拟合的一种方法。
    'log_interval': 20,#跑多少次进行一次日志记录
    'test_interval': 1,#迭代多少次进行一次测试
    'retry': 1,

    ## test path
    'test':['/home/heq/xinyi/workspace/log/NetEase/BGCN_tune/05-27-18-28-54-some_note/1_2918af_Recall@80.pth']
}

