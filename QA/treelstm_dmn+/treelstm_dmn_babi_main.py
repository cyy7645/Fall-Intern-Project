# -*- coding:utf-8 -*-
from treelstm_dmn_babi_loader import BabiDataset, pad_collate
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pickle
from torch.utils.data.dataset import Dataset
from glob import glob
import numpy as np
from torch.utils.data.dataloader import default_collate
from copy import deepcopy as deepcopy
import cyy_dill
import sys
from adict import adict
from Tree import Tree
from TreeLSTM import TreeLSTM
from SentenceTreeLSTM import SentenceTreeLSTM
import constantsF



# 时间记忆模块attention mechanism 的 Attention based GRU
class AttentionGRUCell(nn.Module):
    # 80   80
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        # 80
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal_(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal_(self.U.state_dict()['weight'])

    # GRU结构，对DMN的更改
    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = F.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = F.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h

# 构建attentionGRU的结构，调用AttentionGRUCell类
class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        # 2,4,80
        batch_num, sen_num, embedding_size = facts.size()
        # 80
        C = Variable(torch.zeros(self.hidden_size)).cuda()
        # 0 1 2 3
        for sid in range(sen_num):
            # 对每一个句子 2*1*80
            fact = facts[:, sid, :]
            # g: 2*1
            g = G[:, sid]
            if sid == 0:
                # C: 2*80
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        # 2*80
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal_(self.z1.state_dict()['weight'])
        init.xavier_normal_(self.z2.state_dict()['weight'])
        init.xavier_normal_(self.next_mem.state_dict()['weight'])

    # facts: 2*4*80
    # questions: 2*1*80
    # prevM: 2*1*80
    # 产生标量 gate
    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        # 2*4*80
        # print("facts_size:",facts.size())
        questions = questions.expand_as(facts)
        # 2*4*80
        prevM = prevM.expand_as(facts)
        # 2*4*320
        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)
        # 8 * 320
        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)
        # G: 2*4
        return G

    # Episode Memory Updates，更新memory m
    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        # G: 2*4
        G = self.make_interaction(facts, questions, prevM)
        # 2*80
        C = self.AGRU(facts, G)
        # 去除张量中一维的维度
        # cat 2*80 + 2*80 + 2*80 = 2*240
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        # next_mem: 3*80
        next_mem = F.relu(self.next_mem(concat))
        # next_mem: 3*1*80
        next_mem = next_mem.unsqueeze(1)
        return next_mem

class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        # 80, 80
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    # 把question转换为词向量，再通过gru得到ﬁnal hidden state
    def forward(self, questions, word_embedding):
        '''
        # 假设 batch = 2, token = 3,句子由3个单词组成
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        # 2*3*80
        questions = word_embedding(questions)
        # questions = 2*2*80
        _, questions = self.gru(questions)
        # 交换0,1维的值
        questions = questions.transpose(0, 1)
        return questions


# 接受contexts
class InputModule(nn.Module):
    # 该类继承于nn.Modul，重新定义其构造函数
    def __init__(self, vocab_size, hidden_size):
        # super(InputModule,self) 首先找到 InputModule 的父类（就是类 nn.Module），
        # 然后把类B的对象 InputModule 转换为类 nn.Module 的对象
        # super()函数来调用父类（nn.Module）的init()函数，解决对冲继承问题
        super(InputModule, self).__init__()
        # 除了父类构造函数中的成员变量和成员函数，额外创建新的成员变量和成员函数

        # hidden_size=80
        self.hidden_size = hidden_size
        self.tree_model = TreeLSTM(
             # 2412
            in_dim=80,  # 300
            mem_dim=80,  # 150
            hidden_dim=50,  # 50
            num_classes=5)  # false
        # 双向gru,输入的特征值维度,GRU看做多层神经网络，有多个weight，如layer1.weight layer2.weight
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        # name是成员变量或函数名，param值的是变量本身或函数本身
        # 找到所有的weight，对weight进行初始化
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal_(param)
        self.dropout = nn.Dropout(0.1)

    # 把input转化为词向量，再把词向量通过双向GRU获得fi，Input Fusion Layer
    def forward(self, contexts, word_embedding,tree_dict):
        '''
        contexts.size() -> (#batch, #sentence, #token) 1 4
        word_embedding() -> (#batch, #sentence, #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        '''
        # 1，4，7




        batch_num, sen_num, token_num = contexts.size()

        # 改变context的尺寸，变成 #batch * （#sentence * #token）2*4*8
        # contexts = contexts.view(batch_num, -1)
        # print("contexts:",contexts)
        # 对contexts进行mask
        contexts = contexts.cpu()
        contexts_np = contexts.numpy()
        contexts_list = contexts_np.tolist()
        # print("contexts_list:",contexts_list)
        for batch in range(len(contexts_list)):
            # mask 删除list中的0
            for i in range(len(contexts_list[batch])):
                contexts_list[batch][i] = list(filter(lambda x: x != 0, contexts_list[batch][i]))
            for i in range(len(contexts_list[batch]) - 1, -1, -1):
                if len(contexts_list[batch][i]) == 0:
                    contexts_list[batch] = contexts_list[batch][:-1]
        contexts_list_hidden = []
        # contexts batch * 句子个数 * （tree,list）
        for batch in range(len(contexts_list)):
            # test_batch实例 包含csentences 是个3维 batch * 句子 * 单词 ctrees是个2维 batch * 句子
            # test_batch = SICKDataset(vocab, contexts[batch])

            # 从contexts中拿出tree 和 list
            cid_batch = []
            csent_batch = []
            for sences in range(len(contexts_list[batch])):
                # print("contexts_list[batch][0]: ",contexts_list[batch][sences][0])
                cid_batch.append(contexts_list[batch][sences][0])
                csent = contexts_list[batch][sences][1:]
                csent_batch.append(csent)

            # ctree_batch, csent_batch = test_batch.get()
            # 返回的是一个context中所有句子的hidden
            contexts_batch = self.tree_model.forward(cid_batch, csent_batch,word_embedding,tree_dict)
            # print("contexts_batch:",contexts_batch)
            contexts_list_hidden.append(contexts_batch)

        contexts_list = self.pad_zoro_tensor(contexts_list_hidden)



        contexts = torch.stack(contexts_list, 0).cuda()

        # h0维度 2*2*80, 因为bidirectional所以第一维是2
        h0 = Variable(torch.zeros(2, batch_num, self.hidden_size).cuda())
        # 通过gru对context,h0更新，# 拼接 2, 4, 160 + 2, 2, 80
        facts, hdn = self.gru(contexts,h0)
        # 把facts分割成两部分，为两部分的对应元素相加
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        # print("facts:",facts)
        return facts
    
    def pad_zoro_tensor(self,contexts_list_hidden):
            lengths = []
            for contexts in contexts_list_hidden:
                lengths.append(contexts.size(0))
            max_size = max(lengths)
            # size1 = cstates1.size(0)
            # size2 = cstates2.size(0)
            for i in range(len(contexts_list_hidden)):
                contexts_list_hidden[i] = F.pad(contexts_list_hidden[i], (0, 0, 0, max_size - contexts_list_hidden[i].size(0)), 'constant',
                                         0)
            return contexts_list_hidden

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal_(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)
    # M: 42  questions:80
    # 通过q和m获得answer
    def forward(self, M, questions):
        M = self.dropout(M)
        concat = torch.cat([M, questions], dim=2).squeeze(1)
        z = self.z(concat)
        return z


# 继承于nn.Module，构建模型结构，包含input,question,memory,answer模块
class DMNPlus(nn.Module):
    # hidden_size=80 词向量大小, vocab_size=41 字典大小, num_hop=3 回答模块迭代次数, qa=dset.QA 语料库字典
    def __init__(self, hidden_size, vocab_size, tree_dict,num_hop=3):
        # 函数是用于调用父类(超类)DMNPlus
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.tree_dict=tree_dict
        # sparse=True通常运行速度更快
        # 根据字典大小和嵌入向量大小，把每一个输入转换为嵌入向量大小的tensor
        # 当输入context 4*7时，输出 4*7*80
        #self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True).cuda()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True).cuda()
        # 从均匀分布U(-sqrt(3), sqrt(3))中生成值，填充输入的张量或变量
        # state_dict()是一个字典，存放如weight，所以这一步是初始化weight
        # state_dict (dict) – 保存parameters和persistent buffers的字典。
        # 保存着module的所有状态（state）
        init.uniform_(self.word_embedding.state_dict()['weight'], a=-(3 ** 0.5), b=3 ** 0.5)
        # 目标函数
        self.criterion = nn.CrossEntropyLoss(size_average=False)

        self.input_module = InputModule(vocab_size, hidden_size)
        self.question_module = QuestionModule(vocab_size, hidden_size)
        # 3*1*80
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    # 在get_loss中调用
    def forward(self, contexts, questions):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        '''
        facts = self.input_module(contexts, self.word_embedding, self.tree_dict)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        # 更新三次memory
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)
        preds = self.answer_module(M, questions)
        return preds

    def interpret_indexed_tensor(self, var):
        if len(var.size()) == 3:
            # var -> n x #sen x #token
            for n, sentences in enumerate(var):
                for i, sentence in enumerate(sentences):
                    s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                    print(f'{n}th of batch, {i}th sentence, {s}')
        elif len(var.size()) == 2:
            # var -> n x #token
            for n, sentence in enumerate(var):
                s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print(f'{n}th of batch, {s}')
        elif len(var.size()) == 1:
            # var -> n (one token per batch)
            for n, token in enumerate(var):
                s = self.qa.IVOCAB[token.data[0]]
                print(f'{n}th of batch, {s}')

    def get_loss(self, contexts, questions, targets):
        output = self.forward(contexts, questions)
        loss = self.criterion(output, targets)
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        preds = F.softmax(output)
        _, pred_ids = torch.max(preds, dim=1)
        corrects = (pred_ids.data == answers.data)
        acc = torch.mean(corrects.float())
        return loss + reg_loss, acc

if __name__ == '__main__':
    import sys
    old_stdout = sys.stdout
    # 日志文件地址
    log_file = open("log.txt","w")

    sys.stdout = log_file


    # 训练十次
    for run in range(10):
        # 20个训练任务
        for task_id in range(1,21):
            with open('parsed_dataset/datafile_{}'.format(task_id), 'rb') as f:
                dset = cyy_dill.load(f)
            vocab_size = len(dset.QA.VOCAB)
            hidden_size = 80
            # 实例化预定义的模型
            model = DMNPlus(hidden_size, vocab_size, dset.tree_dict,num_hop=3)
            model.cuda()
            # 把模型放进GPU
            # model.cuda()
            # 记录当前accuracy < best的次数
            early_stopping_cnt = 0
            # 停止的标记
            early_stopping_flag = False
            best_acc = 0
            # 定义优化器
            optim = torch.optim.Adam(model.parameters())
            
            for epoch in range(256):
                dset.set_mode('train')
                train_loader = DataLoader(
                    dset, batch_size=100, shuffle=True, collate_fn=pad_collate
                )

                model.train()
                if not early_stopping_flag:
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(train_loader):
                        optim.zero_grad()
                        contexts, questions, answers = data
                        batch_size = contexts.size()[0]
#                         print("batch_size",batch_size)
                        contexts = Variable(contexts.long().cuda())
                        questions = Variable(questions.long().cuda())
                        answers = Variable(answers.cuda())

                        loss, acc = model.get_loss(contexts, questions, answers)
                        loss.backward()
                        total_acc += acc * batch_size
                        cnt += batch_size

                        if batch_idx % 20 == 0:
                            print(f'[Task {task_id}, Epoch {epoch}] [Training] loss : {loss.data[0]: {10}.{8}}, acc : {total_acc / cnt: {5}.{4}}, batch_idx : {batch_idx}')
                            sys.stdout.flush()
                        optim.step()

                    dset.set_mode('valid')
                    valid_loader = DataLoader(
                        dset, batch_size=100, shuffle=False, collate_fn=pad_collate
                    )

                    model.eval()
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(valid_loader):
                        contexts, questions, answers = data
                        batch_size = contexts.size()[0]
                        contexts = Variable(contexts.long().cuda())
                        questions = Variable(questions.long().cuda())
                        answers = Variable(answers.cuda())

                        _, acc = model.get_loss(contexts, questions, answers)
                        total_acc += acc * batch_size
                        cnt += batch_size

                    total_acc = total_acc / cnt
                    if total_acc > best_acc:
                        best_acc = total_acc
                        best_state = model.state_dict()
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                        if early_stopping_cnt > 20:
                            early_stopping_flag = True

                    print(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}}')
                    sys.stdout.flush()
#                     with open('log17.txt', 'a') as fp:
#                         fp.write(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}}' + '\n')
                    if total_acc == 1.0:
                        break
                else:
                    print(f'[Run {run}, Task {task_id}] Early Stopping at Epoch {epoch}, Valid Accuracy : {best_acc: {5}.{4}}')
                    sys.stdout.flush()
                    break

            dset.set_mode('test')
            test_loader = DataLoader(
                dset, batch_size=100, shuffle=False, collate_fn=pad_collate
            )
            test_acc = 0
            cnt = 0

            for batch_idx, data in enumerate(test_loader):
                contexts, questions, answers = data
                batch_size = contexts.size()[0]
                contexts = Variable(contexts.long().cuda())
                questions = Variable(questions.long().cuda())
                answers = Variable(answers.cuda())

                model.load_state_dict(best_state)
                _, acc = model.get_loss(contexts, questions, answers)
                test_acc += acc * batch_size
                cnt += batch_size
            print(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Test] Accuracy : {test_acc / cnt: {5}.{4}}')
            sys.stdout.flush()

    log_file.close()
