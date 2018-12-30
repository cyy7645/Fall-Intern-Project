# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:56:39 2018

@author: 陈俏均
"""
from torch.utils.data import sampler
import torch
import numpy as np
#import numpy as np
#import bisect


# 定义 DataLoad类，使数据以batch形式进入模型
class DataLoad:
    
    def __init__(self, dataset, batch_size, shuffle = True, drop_last = False):
        # buckets list 根据contexts长度分组
        self.buckets = bucket(dataset)  
        # 打乱 list
        if shuffle:
            np.random.shuffle(self.buckets)
            random_samplers = [sampler.RandomSampler(bucket) for bucket in self.buckets]
        else:
            random_samplers = [sampler.SequentialSampler(bucket) for bucket in self.buckets]
        self.sampler = [sampler.BatchSampler(s, batch_size, drop_last) for s in random_samplers]
        
        
    def __iter__(self):
        return _DataLoad(self) 
        
    def __len__(self):
        return len(self.sampler)

class bucket:
    
    def __init__(self, dataset):
        buckets = {}
        # 对每个 QA 而言
        # contexts [0, 1] 代表每个QA对中句子id
        # questions [10, 11, 1] 单词在字典中的 id
        # [5] 单词在字典中的 id
        for contexts, questions, answers in dataset:
            # 根据每个QA对中 context 的长度分组 0-6 7-12 13-18 ...分别为一组
            num_contexts = len(contexts) // 6
            try:
                buckets[num_contexts]
            except KeyError:
                # {0: [], 1:[], 2:[] ... }
                buckets[num_contexts] = []
            finally:
                buckets[num_contexts].append([contexts, questions, answers])
        # buckets.values() 把每个key(代表对应的组)对应的values形成1个list, 再把不同keys对应的list合并成大lists
        # [ [ [[0, 1], [10, 11, 1], [5]],[[3, 4], [10, 11, 1], [5]] ] , [ [[0, 1, 3, 4, 6, 7, 9, 10], [10, 11, 12], [16]],[[0, 1, 3, 4, 6, 7, 9, 10], [10, 11, 12], [16]] ]]
        # 最外层list长度为 keys的个数 第二层是 key=0,1,2...的qa对数

        self.buckets = list(buckets.values())
        #self.num_buckets = [len(bucket) for bucket in self.buckets]
        #self.num_buckets = np.cumsum(self.num_buckets)

    def __getitem__(self, i):
        return self.buckets[i]
        
        #loc = bisect.bisect_right(self.num_buckets, i)
        #print(loc)
        #if loc != 0:
        #    prev_num = self.num_buckets[loc - 1]
        #else:
        #    prev_num = 0
        #print(prev_num)
        #inner_loc = i - prev_num
        #print(inner_loc)
        #return self.buckets[loc][inner_loc]
    
    def __len__(self):
        return len(self.buckets)
    
    
    def __setitem__(self, i, value):
        self.buckets[i] = value
        
    
class _DataLoad:
    
    def __init__(self, loader):
        self.buckets = loader.buckets
        self.sampler = loader.sampler
        self.sampler_iter = [iter(s) for s in self.sampler]
    
    
    def __next__(self):
        for i, s in enumerate(self.sampler_iter):
            while True:
                try:
                    indices = next(s)
                except StopIteration:
                    break
                else:
                    batch = [self.buckets[i][j] for j in indices]
                    contexts, questions, answers = zip(*batch)
                    questions = torch.LongTensor(questions).cuda()
                    answers = torch.LongTensor(answers).squeeze(dim = 1).cuda()
                    return contexts, questions, answers 
        raise StopIteration
        
    def __len__(self):
        return len(self.sampler)    

    def __iter__(self):
        return self
