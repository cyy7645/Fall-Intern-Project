from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import re
import numpy as np
import copy
from tqdm import tqdm
import pickle
from nltk.parse.stanford import StanfordDependencyParser
from concurrent import futures
from time import time
import time
import dill
import multiprocessing
import copy
import json
from adict import adict
from Tree import Tree

# 解析包路径
path_to_jar = '/data/notebook/jupyterhub/notebook_dirs/chenyy/QA/treelstm_dmn+/lib/stanford-parser/stanford-parser.jar'
path_to_models_jar = '/data/notebook/jupyterhub/notebook_dirs/chenyy/QA/treelstm_dmn+/lib/stanford-parser/stanford-parser-3.9.1-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)


# padding 函数
def pad_collate(batch):
    max_context_sen_len = float('-inf')
    max_context_len = float('-inf')
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _ = elem
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    max_context_len = min(max_context_len, 70)
    for i, elem in enumerate(batch):
        _context, question, answer = elem
        _context = _context[-max_context_len:]
        context = np.zeros((max_context_len, max_context_sen_len))
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, question, answer)
    return default_collate(batch)

# 数据处理类 --> context 转换成 树
class BabiDataset(Dataset):
    def __init__(self, task_id, mode='train'):
        self.mode = mode
        raw_train, raw_test = get_raw_babi(task_id)
        self.QA = adict()
        self.QA.VOCAB = {'<PAD>': 0}
        self.QA.IVOCAB = {0: '<PAD>'}
        self.count_id = 1
        self.tree_dict = {}
        self.train, self.count_id,tree_dict1 = self.get_indexed_qa(raw_train,self.count_id)
        self.tree_dict.update(tree_dict1)
        self.valid = [self.train[i][int(-len(self.train[i])/10):] for i in range(3)]
        self.train = [self.train[i][:int(9 * len(self.train[i])/10)] for i in range(3)]
        self.test, self.count_id, tree_dict2= self.get_indexed_qa(raw_test,self.count_id)
        self.tree_dict.update(tree_dict2)

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        elif self.mode == 'valid':
            return len(self.valid[0])
        elif self.mode == 'test':
            return len(self.test[0])

    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers = self.train
        elif self.mode == 'valid':
            contexts, questions, answers = self.valid
        elif self.mode == 'test':
            contexts, questions, answers = self.test
        return contexts[index], questions[index], answers[index]
    
    # 通过对句子解析产生的中间变量line(包含单词的父子关系)，构建树结构
    def read_tree(self,line):
        # print("line:",line)
        # 3 3 5 5 11 5 6 9 6 11 0 11
        # parents = list(map(int, line.split()))
        parents = line
        trees = dict()
        root = None
        # 1 - 13
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i  # 1  2  7  8
                prev = None
                while True:
                    parent = parents[idx - 1]  # parent = 3  5  11  0 --- 3 5  11 -- 6  9
                    if parent == -1:
                        break
                    tree = Tree()  # 实例化
                    if prev is not None:
                        tree.add_child(
                            prev)  # null  5这棵树的孩子结点的3   11这棵树的孩子结点为5  0这棵树的孩子结点为11  5这棵树的孩子结点为3  11这棵树的孩子结点为5
                    trees[
                        idx - 1] = tree  # trees[0]为空Tree()  trees[5]为 5这棵树   trees[4]为 11这棵树  trees[10]为 0这棵树  trees[1]为 3这棵树 trees[2]为 5这棵树 trees[6]为 6这棵树 trees[7]为 9这棵树
                    # tree.idx为树的编号
                    tree.idx = idx - 1  # 0  2  4  10  1  2  -- 6  7
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree  # 0这棵树是root
                        break
                    else:
                        prev = tree  # prev为 Tree()  tree.idx = 0  prev= 5这棵树   prev= 11这棵树  prev= 0这棵树 prev= 3这棵树   5这棵树  9
                        idx = parent  # idx = 3   5  11  0   3  5
        return root
        ###################################

    def get_indexed_qa(self, raw_babi,count_id):
        unindexed,final_count_id,final_tree_dict = self.get_unindexed_qa(raw_babi,count_id)


        def parallel(qa):
            # qa 是一个字典
            # {'C': ['21 John travelled to the hallway . ', '22 Mary journeyed to the bathroom . '], 'Q': 'Where is John', 'A': 'hallway', 'S': [0]}
            tree_id = []
            context = [c.lower().split() for c in qa['C']]
            # print(context)
            # 对context中的每个句子
            for con in context:
                # tree_id: [21,22]
                tree_id.append(int(con[0]))
                for token in con[1:]:
                    self.build_vocab(token)
            # 二维list 单词变成数字
            context = [[self.QA.VOCAB[token] for token in sentence[1:]] for sentence in context]
            # 把每个list的第一位放 tree对应的id
            for i in range(len(context)):
                context[i].insert(0,tree_id[i])
            # tree_id=[]
            question = qa['Q'].lower().split()

            for token in question:
                self.build_vocab(token)
            question = [self.QA.VOCAB[token] for token in question]

            self.build_vocab(qa['A'].lower())
            answer = self.QA.VOCAB[qa['A'].lower()]

            # print("context: ",context)
            # print("question: ", question)
            # print("answer: ", answer)

            # context: [[1, 1, 2, 3, 4, 5, 6], [2, 7, 8, 3, 4, 9, 6]]
            # question: [10, 11, 1]
            # answer: 5  都代表字典中对应的元素
            return context, question, answer

        with futures.ThreadPoolExecutor(20) as executor:

            res = executor.map(parallel, unindexed)

        contexts = []
        questions = []
        answers = []
        for i, result in enumerate(res):

            # print("result: ", result)
            (context, question, answer) = result
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
            # br

        return (contexts, questions, answers),final_count_id,final_tree_dict



    # 根据原始字符串获得task list
    def get_unindexed_qa(self, raw_babi, count_id):
        tasks = []
        task = None
        babi = raw_babi.strip().split('\n')
        # babi ['1 Mary moved to the bathroom.', '2 John went to the hallway.', '3 Where is Mary? \tbathroom\t1',...]
        # print("babi: ",babi)
        # parse_count_id = count_id
        context_count_id = count_id

        # 对context中的每个句子多线程解析
        def parallel(line):
            if line.find('?') == -1:
                line = line.strip()
                line = line[line.find(' ') + 1:]
                result, = dependency_parser.raw_parse(line)
                # parse_count_id += 1
                res = result.to_conll(4)
                nums = re.findall(r"\d+\.?\d*", res)
                results = list(map(int, nums))
                # print("results",results)
                root = self.read_tree(results)
                return root
            else:
                return 'f'

        with futures.ThreadPoolExecutor(20) as executor:
            tree_dict = {}
            r = {}
            count_all = count_id
            for i, line in enumerate(babi):
                if line.find('?') == -1:
                    future = executor.submit(parallel,line)
                    # print("future:",future)
                    tree_dict[future] = count_all
                    count_all += 1

            done_iter = futures.as_completed(tree_dict)

            # print("done_iter:",done_iter)
            for future in done_iter:
                try:

                    idx = tree_dict[future]
                    res = future.result()
                    if res != 'f':
                        r[idx] = res
                except:
                    pass

        for i, line in enumerate(babi):
            id = int(line[0:line.find(' ')])
            if id == 1:
                task = {"C": "", "Q": "", "A": "", "S": ""}
                counter = 0
                id_map = {}

            line = line.strip()
            line = line.replace('.', ' . ')
            line = line[line.find(' ') + 1:]
            # if not a question
            if line.find('?') == -1:
                # context_count_id += 1
                task["C"] += str(context_count_id) + ' ' + line + '<line>'
                id_map[id] = counter
                counter += 1
                context_count_id += 1
            else:
                idx = line.find('?')
                tmp = line[idx + 1:].split('\t')
                task["Q"] = line[:idx]
                task["A"] = tmp[1].strip()
                task["S"] = []  # Supporting facts
                for num in tmp[2].split():
                    task["S"].append(id_map[int(num.strip())])
                tc = task.copy()
                tc['C'] = tc['C'].split('<line>')[:-1]
                tasks.append(tc)
        # [{'C': ['21 John travelled to the hallway . ', '22 Mary journeyed to the bathroom . '], 'Q': 'Where is John', 'A': 'hallway', 'S': [0]}, ...]
        # print('tasks[0:2]: ', tasks)
        return tasks, count_all, r

    # 通过传进来的 单词 构建QA字典
    def build_vocab(self, token):
        if not token in self.QA.VOCAB:
            next_index = len(self.QA.VOCAB)
            self.QA.VOCAB[token] = next_index
            self.QA.IVOCAB[next_index] = token


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

def build_vocab(raw_babi):
    lowered = raw_babi.lower()
    tokens = re.findall('[a-zA-Z]+', lowered)
    types = set(tokens)
    return types



if __name__ == '__main__':
    start = time.time()
    # 对20个tasks进行解析，把解析的文件存在parsed_dataset文件夹中
    for i in tqdm(range(1,21)):
        dset_train = BabiDataset(i)
        with open('parsed_dataset/datafile_'+str(i), 'wb') as f:
            dill.dump(dset_train, f)
        print('achieve: '+str(i))
    end = time.time()
    print("time:",end - start,"s")
    print(dset_train.QA.VOCAB)
    # with open('/Users/cyy7645/Desktop/datafile_1', 'rb') as f:
    #     dset_train = dill.load(f)
    # print(dset_train.QA.VOCAB)
    # print(dset_train.test[0])
    # # print(len(dset_train.test[0])+len(dset_train.train[0])+len(dset_train.valid[0]))
    # print(dset_train.tree_dict)
    # print(dset_train.QA.VOCAB)
    # print(len(dset_train.train[0]))
    # print(dset_train.train[0])
    # print(len(dset_train.train[0][1]))

