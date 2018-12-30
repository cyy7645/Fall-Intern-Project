# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:49:29 2018

@author: 陈俏均
"""

# 定义常量

embed_dim = 80      # 词向量维度
mem_dim = 80         # GRU，TreeLSTM隐藏层维度
sparse = False        # Embedding 层参数，权重矩阵的梯度是否是一个稀疏张量，会影响优化算法的选择
input_dropout = 0.5   # Input Module GRU之后的 dropout层参数
output_dropout = 0.1  # Answer Module 中对memory的 dropout层参数
update_times = 3     # Episode Memory Updates 中 memory update的次数
weight_decay = 0.001  # 优化算法Adam的权重衰减参数
batch_size = 100    # 每次进入模型的batch
num_runs = 10     # 运行模型的次数 排除模型不稳定性影响
task_num = 21   # 任务总数
num_epoch = 256   
#lr = 0.00001
