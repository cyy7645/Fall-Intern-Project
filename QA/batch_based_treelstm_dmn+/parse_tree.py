# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:51:20 2018

@author: 陈俏均
"""

# 数据预处理代码，包括对contexts(输入内容)中的每个句子进行解析，形成中间文件储存在本地
from glob import glob
from torch.utils.data.dataset import Dataset
from tree import get_line, read_tree
from concurrent import futures
import pickle
import time
from tqdm import tqdm


MAX_WORKERS = 20

class Babidataset(Dataset):
    # 原始数据集路径
    PATH = '/data/notebook/jupyterhub/notebook_dirs/chenyy/QA/data//en-10k/qa{}_*'
    # 解析完成后的储存路径
    SAVE_PATH = 'parsed_data/qa{}_save'
    
    
    def __init__(self, task_id, mode = 'train', valid_split = 0.9):      
        self.task_id = task_id
        self.valid_split = valid_split
        self._mode = mode
        self.vocab = {'<PAD>' : 0}
        self.ivocab = {0 : '<PAD>'}
        # 分别是个字符串
        raw_train, raw_test = get_raw_babi(self.PATH, self.task_id)
        # train存context question answer对应的字典id  train_tree_dict字典 id:tree
        self.train, self.train_tree_dict = get_raw_qa(self.vocab, self.ivocab, raw_train)
        self.valid = [self.train[i][int(valid_split * len(self.train[i])):] for i in range(3)]
        self.train = [self.train[i][:int(valid_split * len(self.train[i]))] for i in range(3)]
        self.test, self.test_tree_dict = get_raw_qa(self.vocab, self.ivocab, raw_test)
        with open(self.SAVE_PATH.format(task_id), 'wb') as f:
                pickle.dump(self, f)

    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, mode):
        self._mode = mode
        
    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers = self.train
        if self.mode == 'valid':
            contexts, questions, answers = self.valid
        if self.mode == 'test':
            contexts, questions, answers = self.test
        return contexts[index], questions[index], answers[index]
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        if self.mode == 'valid':
            return len(self.valid[0])
        if self.mode == 'test':
            return len(self.test[0])

# 更新字典
def update_vocab(vocab, ivocab, token):
    if token not in vocab:
        next_idx = len(vocab)
        vocab[token] = next_idx
        ivocab[next_idx] = token
    else:
        next_idx = vocab[token]
    return next_idx

# 获得train, test数据集的字符串
def get_raw_babi(path, task_id):
    paths = glob(path.format(task_id))
    for path in paths:
        if 'train' in path:
            with open(path, 'r') as f:
                train = f.read()
        if 'test' in path:
            with open(path, 'r') as f:
                test = f.read()
    return train, test

def get_raw_qa(vocab, ivocab, raw_babi):
    contexts = []
    questions = []
    answers = []
    tree_dict = {}
    # 是一个list 每个元素是一个句子
    pairs = raw_babi.strip().split('\n')
#     print("len(pairs): ",len(pairs))
    # pairs = tqdm(pairs)
    with futures.ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
        before = {}
        after = {}
        for i, line in enumerate(pairs):
            # 多线程执行 process 函数
            future = executor.submit(process, line, vocab, ivocab)
            before[future] = i
        done_iter = futures.as_completed(before)
        done_iter = tqdm(done_iter, total=len(pairs))
        for future in done_iter:
            after[before[future]] = future.result()

    for i in range(len(after)):
        content = after[i]
        ID = content[0]
        if ID == 1:
            context = []
        # 是context
        if len(content) == 2:
            #contexts  append的是句子的编号
            context.append(i)
            tree_dict[i] = content[1]
        # 是 question 和 answer
        elif len(content) == 3:
            #question and answer
            questions.append(content[1])
            answers.append(content[2])
            f_context = context.copy()
            contexts.append(f_context)
        # contexts 二维list 每个元素是每个context的句子编号  list长度为QA对个数
        #  questions 二维list 每个元素是句子中单词id
        # answers 二维list 每个元素是单词id
#     print("len(contexts): ",len(contexts))
#     print("contexts: ",contexts)
#     print("questions: ", questions)
#     print("answers: ", answers)
    return (contexts, questions, answers), tree_dict

# 把context中的每个单词转变为字典中的id
def process_to_idx(context, vocab, ivocab):
    context = [update_vocab(vocab, ivocab, token) for token in context.lower().split()]
    return context    

# 为每个句子生成一颗树
def parse_tree(context, context_idx):
    line = get_line(context)
    root = read_tree(line, context_idx)
    return root

# 调用解析包把句子解析成依存树
def process(line, vocab, ivocab):
    ID = int(line[:line.find(' ')])
    line = line[(line.find(' ') + 1):]
    line = line.strip()
    line = line.replace('.', ' .')
    if line.find('?') == -1:
        context_idx = process_to_idx(line, vocab, ivocab)
        root = parse_tree(line, context_idx)
#         print(root.depth)
        return ID, root
    else:
        loc = line.find('?')
        question = process_to_idx(line[:loc], vocab, ivocab)
        temp = line[(loc + 1):].split('\t')
        answer = process_to_idx(temp[1].strip(), vocab, ivocab)
        # ID句子编号  question和answer是字典的id
        return ID, question, answer
# SAVE_PATH = './data/en-10k/qa0_save'
# start = time.time()
# dataset = Babidataset(3)
# end = time.time()
# print("time:",end - start,"s")
# SAVE_PATH = "/Users/cyy7645/Desktop/qa3_save"
# with open(SAVE_PATH, 'rb') as f:
#     dset_train = pickle.load(f)
#
# for i in range(20):
#     print(dset_train.train_tree_dict[dset_train[1][0][i]].depth)
