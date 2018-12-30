# -*- coding:utf-8 -*-
# 用于处理输入，从每一个训练文档或测试文档中提取出context,question,answer，并用数字替换单词
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import re
import numpy as np

# 字典类，继承于dict
# 该类用于储存两个字典 分别为数据集中所有单词的 key:value 和 value:key
class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

# 对一个批次中的contexts 和 questions 进行padding
def pad_collate(batch):
    max_context_sen_len = float('-inf')
    max_context_len = float('-inf')
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _ = elem
         # 找到一个批次中 context 最长句子的长度（单词个数）
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        # 找到一个批次中 question最长的长度（单词个数）
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        # 找到一个批次中 context 包含的最多句子个数
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    # 正常情况下是全部的句子，当max_context_len>70时，只取前面70个句子
    max_context_len = min(max_context_len, 70)
    for i, elem in enumerate(batch):
        _context, question, answer = elem
        _context = _context[-max_context_len:]
        context = np.zeros((max_context_len, max_context_sen_len))
        # 对小于最长长度的句子后面用0填补
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        # 对小于最长长度的问题后面用0填补
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, question, answer)
    return default_collate(batch)

class BabiDataset(Dataset):
    def __init__(self, task_id, mode='train'):
        # train or validation or test
        self.mode = mode
        # 得到train,test的元素数据，为一个string, 里面包含.txt中的全部内容
        raw_train, raw_test = get_raw_babi(task_id)
        # 创建一个字典类
        self.QA = adict()
        # 类中包含两个字典 key value 相反
        self.QA.VOCAB = {'<PAD>': 0, '<EOS>': 1}
        self.QA.IVOCAB = {0: '<PAD>', 1: '<EOS>'}
        # train为一个元组，内部包含3个部分，分别为问题，答案和内容，都是数组，数组元素为Int
        # question为二维数组，contexts和answers为一维数组，内部元素都是数字
        self.train = self.get_indexed_qa(raw_train)
        # 对train中的每个部分划分valid和train，10%：90%
        self.valid = [self.train[i][int(-len(self.train[i])/10):] for i in range(3)]
        self.train = [self.train[i][:int(9 * len(self.train[i])/10)] for i in range(3)]
        self.test = self.get_indexed_qa(raw_test)

    def set_mode(self, mode):
        self.mode = mode

    # 返回 问答对 对数
    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        elif self.mode == 'valid':
            return len(self.valid[0])
        elif self.mode == 'test':
            return len(self.test[0])
    
    # 设置 mode
    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers = self.train
        elif self.mode == 'valid':
            contexts, questions, answers = self.valid
        elif self.mode == 'test':
            contexts, questions, answers = self.test
        return contexts[index], questions[index], answers[index]

    def get_indexed_qa(self, raw_babi):
        # 为一个list，每个元素是一个字典，字典里面储存了每个问答对的C,Q,A,S
        unindexed = get_unindexed_qa(raw_babi)
        questions = []
        contexts = []
        answers = []
        for qa in unindexed:
            context = [c.lower().split() + ['<EOS>'] for c in qa['C']]

            for con in context:
                for token in con:
                    self.build_vocab(token)
            context = [[self.QA.VOCAB[token] for token in sentence] for sentence in context]
            question = qa['Q'].lower().split() + ['<EOS>']

            for token in question:
                self.build_vocab(token)
            question = [self.QA.VOCAB[token] for token in question]

            self.build_vocab(qa['A'].lower())
            answer = self.QA.VOCAB[qa['A'].lower()]


            contexts.append(context)
            questions.append(question)
            answers.append(answer)
        return (contexts, questions, answers)
    
    # 把每个单词添加到字典中
    def build_vocab(self, token):
        if not token in self.QA.VOCAB:
            next_index = len(self.QA.VOCAB)
            self.QA.VOCAB[token] = next_index
            self.QA.IVOCAB[next_index] = token

# 加载每个任务train,test集的全部内容
def get_raw_babi(taskid):
    paths = glob('../data/en-10k/qa{}_*'.format(taskid))
    for path in paths:
        if 'train' in path:
            with open(path, 'r') as fp:
                train = fp.read()
        elif 'test' in path:
            with open(path, 'r') as fp:
                test = fp.read()
    return train, test

# 返回包含唯一单词的 set
def build_vocab(raw_babi):
    lowered = raw_babi.lower()
    tokens = re.findall('[a-zA-Z]+', lowered)
    types = set(tokens)
    return types


def get_unindexed_qa(raw_babi):
    tasks = []
    task = None
    # 把原始数据使用\n分割，此时babi为一个list，包含字符串，每个字符串为一个string
    babi = raw_babi.strip().split('\n')
     # 第i个句子line
    for i, line in enumerate(babi):
        # 按空格划分，第一位是数字 eg.1 2 3，作为id
        id = int(line[0:line.find(' ')])
        # 如果id为1，是新的问答对
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": ""}
            counter = 0
            id_map = {}
        # 去掉每个句子的前后空格
        line = line.strip()
        # 把标点符号.和前后的单词分开 eg.'1 Mary moved to the bathroom . '
        line = line.replace('.', ' . ')
        # 去掉索引 eg. 'Mary moved to the bathroom . '
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            # task["C"]为string，由原来的""变为 'Mary moved to the bathroom . <line>'
            task["C"] += line + '<line>'
            id_map[id] = counter
            counter += 1
        else:
            idx = line.find('?')
            # list存放answer和位置 eg.[' ', 'bathroom', '8']
            tmp = line[idx+1:].split('\t')
            # task:{'A': 'bathroom','C': 'Mary moved to the bathroom . <line>',
            # #'Q': '15 Where is Sandra','S': ''}
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = [] # Supporting facts
            for num in tmp[2].split():
                # 把原来从1开始的句子编号转变为从0开始
                task["S"].append(id_map[int(num.strip())])
            tc = task.copy()
            tc['C'] = tc['C'].split('<line>')[:-1]
            tasks.append(tc)
            # tasks为一个list,储存元素为字典，每个字典保存了问答对的C,Q,A,S
            # [{'A': 'bathroom',
            #  'C': ['Mary moved to the bathroom . ', 'John went to the hallway . '],
            #  'Q': 'Where is Mary',
            #  'S': [0]},...]
    return tasks

# 测试代码
if __name__ == '__main__':
    dset_train = BabiDataset(20, is_train=True)
    train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, collate_fn=pad_collate)
    for batch_idx, data in enumerate(train_loader):
        contexts, questions, answers = data
        break
