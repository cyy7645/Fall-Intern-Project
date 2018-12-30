# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:32:03 2018

@author: 陈俏均
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from parse_tree import Babidataset
from dataload import DataLoad
import pickle
import constant

def create_zero_tensor(chunks, *args):
    child_ch = Variable(torch.zeros(*args)).cuda()
    result = child_ch.chunk(chunks, dim = child_ch.dim() -1)
    return result

# 找到一棵树中的child结点和leaf结点,添加到 fold中,对一批次内所有contexts执行此函数 
def encode_tree_fold(fold, tree):

    def encode_node(node):
        if node.num_child == 0:
            # 每棵树中的每个结点都有id
            return fold.add('leaf', node.idx)
        else:
            child_c, child_h = zip(*map(lambda x: encode_node(x), node.child))
            return fold.add('children', node.idx, child_c, child_h)

    encoding, _ = encode_node(tree)

    # fold.leaf in fold : [14, 3, 4, 14, 3, 4, 1, 3, 4, 12, 3, 4, 7, 3, 4, 7, 3, 4, 1, 3, 4, 7, 3, ...
    # children in fold: 3维list 第一维度是树层数 = 2  第二维度：每一层的孩子结点集合 第三维度：孩子id和它的左右孩子信息 [5, [0, 0, 1], [0, 0, 2]]
    # 返回根结点的 [depth, loc, idx]
    return encoding

# 存放一个批次内所有contexts中每个句子的孩子结点和叶子结点
# 目的是为了生成 leaf一维列表 children四维列表
class TorchFold:
        
    def __init__(self):
        self.leaf = []
        self.children = []
        self.num_child = []
    def add(self, op, *args):
        # *args: 叶结点id （即叶节点代表的单词在字典中的id）
        if op == 'leaf':
            for arg in args:
                self.leaf.append(arg)
            # idx 对叶子结点计数
            idx = len(self.leaf) - 1
            depth = 0
            loc = 0
        # *args: node.idx, child_c, child_h
        # child_c:  ([0, 153], [0, 154])   child_h:  ([0, 153], [0, 154])
        if op == 'children':            
            depth = self.get_depth(args[1])
            dep_index = [args[0]]
            for arg in args[1]:
                dep_index.append(arg)
            num_child = len(args[1])
            try:
                self.children[depth - 1]
            except IndexError:
                self.children.append([])
            try:
                self.num_child[depth - 1]
            except IndexError:
                self.num_child.append({})
            finally:
                if num_child in self.num_child[depth - 1]:
                    loc = self.num_child[depth - 1][num_child]
                else:
                    loc = self.num_child[depth - 1][num_child] = len(self.children[depth - 1])
                    self.children[depth - 1].append([])
                self.children[depth - 1][loc].append(dep_index)
                idx = len(self.children[depth - 1][loc]) - 1
        # 叶子结点 [0,0,idx]  孩子结点[所在的深度,所在的batch,单词id]
        return [depth, loc, idx], [depth, loc, idx]  
    
    # 获得孩子结点的深度  1或2
    # args:([0, 153], [0, 154])
    def get_depth(self, args):
        return sorted(args, key = lambda x: x[0])[-1][0] + 1

# 做 batch 的计算
class unfold(nn.Module):
    
    def __init__(self, nn):
        super(unfold, self).__init__()
        self.nn = nn
    
    def gather(self, c_pool, h_pool, dep_rela):
        #shape of c_pool:[1, num_ops, mem_dim]
        #shape of h_pool:[1, num_ops, mem_dim]
        #shape of dep_rela:[num_ops, 1 + size_of_max_child] [word_idx, ....]
        #mask != -1
        child_c = []
        child_h = []
        word_idx = torch.LongTensor(dep_rela[:, 0]).cuda()
        for i in range(1, dep_rela.shape[-1]):
            rela_idx = torch.LongTensor(dep_rela[:, i]).cuda()
            child_c_one = c_pool[:, rela_idx, :]
            child_h_one = h_pool[:, rela_idx, :]
            child_c.append(child_c_one)
            child_h.append(child_h_one)
        return word_idx, torch.cat(child_c, dim = 0), torch.cat(child_h, dim = 0)

    # 重新整理fold.children列表 按照所在层数和所在的batch储存孩子结点的编号，
    # e.g. [array([[ 2,  1, 38],[ 8,  4, 39]]),array([[ 8,  7,  8, 40]])] 在表其中一层，把孩子数为2和3的放在不同的array中，list中的元素分别代表 当前结点id 和其孩子结点id
    def _pad(self, fold):
        # 遍历fold中的所有孩子结点
        for depth, dep_relas in enumerate(fold.children):
            # 对每个batch
            for i, rela in enumerate(dep_relas):
                new_rela = []
                # 对该批次里的所有孩子
                for child in rela: 
                    new_child = []
                    # new_child 储存孩子结点的编号
                    new_child.append(child[0])
                    for d_depth, loc, idx in child[1:]:
                        new_child.append(sum(fold.num_ops_record[:d_depth]) + sum(fold.num_child_ops[d_depth][:loc]) + idx + 1)
                    new_rela.append(new_child)
                fold.children[depth][i] = np.asarray(new_rela)
        return fold

    # contexts_idx储存所有批次中的句子编号
    def forward(self, fold, contexts_idx, embed, max_num_setence):
        # 此时fold中的children 第一维度为所在的层数，第二维表示该层数所有的孩子结点，根据结点孩子个数不同形成list
        fold = self._pad(fold)
        # 转换为1维tensor
        leaf_word_idx = torch.LongTensor(fold.leaf).cuda()
        c, h = create_zero_tensor(2, 1, 1, 2 * self.nn.mem_dim)
        # 对leaf结点做 embedding
        c_op, h_op = self.nn.leaf(embed, leaf_word_idx)
        # 对所有叶子结点的 cell 和 hidden 拼接
        c = torch.cat([c, c_op], dim = 1)
        h = torch.cat([h, h_op], dim = 1)
        for dep_rela in fold.children:
            # 对每一层有相同孩子个数的结点
            for dep_child in dep_rela:
                word_idx, child_c, child_h = self.gather(c, h, dep_child)
                c_op, h_op = self.nn.child(embed, word_idx, child_c, child_h)
                c = torch.cat([c, c_op], dim = 1)
                h = torch.cat([h, h_op], dim = 1)          
        return self.form_batch(c, h, contexts_idx, max_num_setence)

    # 找到对应batch中的根结点的隐藏层状态，拼接
    def form_batch(self, c, h, contexts_idx, max_num_setence):
        mem_dim = c.size()[-1]
        batch_c = []
        batch_h = []
        for context_idx in contexts_idx:
            idx = torch.LongTensor(context_idx).cuda()
            num_pad = max_num_setence - len(context_idx)
            context_c, context_h = c[:, idx, :], h[:, idx, :]
            if num_pad != 0:
                zero_c, zero_h = create_zero_tensor(2, 1, num_pad, 2 * mem_dim)
                context_c = torch.cat([context_c, zero_c], dim = context_c.dim()-2) 
                context_h = torch.cat([context_h, zero_h], dim = context_h.dim()-2)
            batch_c.append(context_c)
            batch_h.append(context_h)
        return torch.cat(batch_c, dim = 0), torch.cat(batch_h, dim = 0)

# treelstm结构
class treelstm(nn.Module):
    
    def __init__(self, in_dim, mem_dim):
        super(treelstm, self).__init__()        
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        
    #shape inputs: [1, num_ops_step, embedding]
    #shape child_c: [num_child, num_ops_step, mem_dim]    
    #shape child_h: [num_child, num_ops_step, mem_dim]    
    def forward(self, inputs, child_c, child_h):
        #shape child_h:[1, num_ops_step, mem_dim]
        child_h_sum = torch.sum(child_h, dim = 0, keepdim = True)        
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        #shape i,o, h:[1, num_ops_step, mem_dim]
        i, o, u = iou.chunk(3, dim = iou.dim() - 1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u) 
        #shape f:[num_child, num_ops_step, mem_dim]
        f = F.sigmoid(self.fx(inputs).expand_as(child_h) + self.fh(child_h))
        fc = torch.sum(f * child_c, dim = 0, keepdim = True)
        c = i * u + fc
        h = o * F.tanh(c)
        return c, h


class spin(nn.Module):

    def __init__(self, in_dim, mem_dim):        
        super(spin, self).__init__()
        self.mem_dim = mem_dim
        self.treelstm = treelstm(in_dim, mem_dim)

    def _get_inputs(self, embed, word_idx):
        word_idx = word_idx.view((1, -1))
        #shape inputs: [1, num_ops_step, embedding]
        inputs = embed(word_idx)
        return inputs
        
    def leaf(self, embed, word_idx):
        inputs = self._get_inputs(embed, word_idx)
        shape = inputs.size()
        c,  = create_zero_tensor(1, *shape[:-1], self.mem_dim)

        return c, inputs
        
    def child(self, embed, word_idx, child_c, child_h):
        inputs = self._get_inputs(embed, word_idx)
        c, h = self.treelstm(inputs, child_c, child_h)
        return c, h 
    
class InputModule(nn.Module):
    
    def __init__(self, in_dim, mem_dim, input_dropout = 0.1):
        super(InputModule, self).__init__()
        self.mem_dim = mem_dim
        cell = spin(in_dim, mem_dim)
        self.unfold = unfold(cell)
        self.gru = nn.GRU(mem_dim, mem_dim, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(input_dropout)

    # contexts tuple 含batch 每个元素是一个QA对的context中的句子索引
    def forward(self, embed, contexts, tree_dict):
        # 类 存放 leaf 和 child
        fold = TorchFold()
        # 三维 1维[句子深度，树编号] 二维[一个context中的句子] 三维[batch]
        result = []
        max_num_setence = 0
        # context: [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
        for context in contexts:
            context_idx = []
            num_setence = len(context)
            # 记录一个batch中最长的contexts 句子个数
            if num_setence > max_num_setence:
                max_num_setence = num_setence
            # 0 表示标号为0的句子
            for setence in context:
                # 返回每课树的[深度，所在的batch，树的编号]
                context_idx.append(encode_tree_fold(fold, tree_dict[setence]))
            # [[[2, 0, 0],[2, 0, 1],..,], ] 第一维度代表batch
            result.append(context_idx)
        # 记录每一层的结点个数，包括叶子结点
        fold.num_ops_record = [len(fold.leaf)]
        # 记录每一个孩子结点的个数，底层是叶子结点，所以设为0
        fold.num_child_ops = [[0]]
        # dep 表示孩子结点所在的深度
        for dep in fold.children:
            # 该深度下孩子结点的个数
            c_ops = [len(d) for d in dep]
            fold.num_child_ops.append(c_ops)
            fold.num_ops_record.append(sum(c_ops))
        new_result = []
        # new_result只保存树的编号 实现batch的累加 batch=2的第一棵树为 batch=1的最后一个数+1 [[205, 206, 207, 208, 209, 210], [211, 212, 213, 214, 215, 216, 217, 218], [219,..]]
        for context in result:
            new_result.append([sum(fold.num_ops_record[:depth]) + sum(fold.num_child_ops[depth][:loc]) + idx + 1 for depth, loc, idx in context])
        _, encode_context = self.unfold(fold, new_result, embed, max_num_setence)
        #encode_context.size:[batch_size, num_sequence, mem_dim]
        #encode_context.size:[num_sequence, batch_size, mem_dim]
        facts, _ = self.gru(encode_context)
        #facts.size:[batch_size, num_sequence, 2*mem_dim]
        facts = facts[:, :, :self.mem_dim] + facts[:, :, self.mem_dim:]
        return facts
    
class AttentionGRUCell(nn.Module):
    
    def __init__(self, in_dim, mem_dim):
        super(AttentionGRUCell, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.rux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ruh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        
    def forward(self, context, hidden, g):
        #context.size: [batch_size, in_dim]
        #hidden.size: [batch_size, mem_dim]
        #g: [batch_size,]
        rx, ux = self.rux(context).chunk(2, dim = context.dim() - 1)
        rh, uh = self.ruh(hidden).chunk(2, dim = hidden.dim() - 1)
        r = F.sigmoid(rx + rh)
        h_tilda = F.tanh(ux + r * uh)
        g = g.unsqueeze(dim = 1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * hidden
        return h

class AttentionGRU(nn.Module):
    
    def __init__(self, in_dim, mem_dim):
        super(AttentionGRU, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.gru_cell = AttentionGRUCell(self.in_dim, self.mem_dim)
        
    def forward(self, contexts, G):
        #contexts.size: [batch_size, num_setence, embed_size]
        #G.size: [batch_size, num_setence]
        batch_size, num_setence, _ = contexts.size()
        h,  = create_zero_tensor(1, batch_size, self.mem_dim)
        for sidx in range(num_setence):
            context = contexts[:, sidx, :]
            g = G[:, sidx]
            h = self.gru_cell(context, h, g)
        return h

class EpisodicMemory(nn.Module):
    
    def __init__(self, mem_dim):
        super(EpisodicMemory, self).__init__()
        self.mem_dim = mem_dim
        self.attention = AttentionGRU(self.mem_dim, self.mem_dim)
        self.w_1 = nn.Linear(4 * self.mem_dim, self.mem_dim)
        self.w_2 = nn.Linear(self.mem_dim, 1)
        self.w_u = nn.Linear(3 * self.mem_dim, self.mem_dim)
        
    def interact(self, contexts, question, last_memory):
        #contexts.size: [batch_size, num_setence, embed_size]
        #question.size: [batch_size, embed_size]
        #memory.size:   [batch_size, embed_size]
        if question.dim() == 2:
            question = question.unsqueeze(dim = 1).expand_as(contexts)
        if last_memory.dim() == 2:
            last_memory = last_memory.unsqueeze(dim = 1).expand_as(contexts) 
        z = torch.cat([contexts * question,
             contexts * last_memory,
             torch.abs(contexts - question),
             torch.abs(contexts - last_memory)], dim = contexts.dim() - 1)
        G = self.w_2(F.tanh(self.w_1(z))).squeeze(dim = contexts.dim() - 1)
        #G.size:[batch_size, num_setence]
        G = F.softmax(G, dim = 1)
        #print(torch.sum(G, dim = 1))
        #print(G)
        return G
    
    def forward(self, contexts, question, last_memory):
        #contexts.size: [batch_size, num_setence, embed_size]
        #question.size: [batch_size, embed_size]
        #last_memory.size: [batch_size, embed_size]
        G = self.interact(contexts, question, last_memory)
        h = self.attention(contexts, G)
        #h.size: [batch_size, embed_size]
        update = torch.cat([h, question, last_memory],
                            dim = h.dim() - 1)
        memory = F.relu(self.w_u(update))
        return memory


class QuestionModule(nn.Module):
    
    def __init__(self, in_dim, mem_dim):
        super(QuestionModule, self).__init__()
        self.mem_dim = mem_dim
        self.gru = nn.GRU(in_dim, mem_dim, batch_first = True)
    
    def forward(self, embed, questions):
        #quesstion.size: [batch, num_words]
        questions = embed(questions)
        #quesstion.size: [batch, num_words, in_dim]
        _, questions = self.gru(questions)
        #questions.size: [batch, mem_dim]
        questions = questions.transpose(0, 1)
        return questions.squeeze(dim = 1)
 
    
class AnswerModule(nn.Module):
    
    def __init__(self, mem_dim, out_dim, output_dropout = 0.1):
        super(AnswerModule, self).__init__()
        self.o = nn.Linear(2 * mem_dim, out_dim)
        self.dropout = nn.Dropout(output_dropout)
        
    def forward(self, questions, last_memory):
        #questions.size:[batch, mem_dim]
        #last_memory.size:[batch, mem_dim]
        last_memory = self.dropout(last_memory)
        In = torch.cat([questions, last_memory], dim = questions.dim() - 1) 
        Out = self.o(In)
        return Out
    
class DMN(nn.Module):
    
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 mem_dim,
                 sparse = True,
                 input_dropout = 0.1,
                 output_dropout = 0.1,
                 update_times = 3):
        super(DMN, self).__init__()
        self.update_times = update_times
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx = 0, sparse = sparse).cuda()
        init.uniform_(self.embed.state_dict()['weight'], a = -(3**0.5), b = 3**0.5)
        self.input_module = InputModule(embed_dim, mem_dim, input_dropout)
        self.question_module = QuestionModule(embed_dim, mem_dim)
        self.memory = EpisodicMemory(mem_dim)
        self.answer = AnswerModule(mem_dim, vocab_size, output_dropout)
       
    def forward(self, batch_contexts, batch_questions, tree_dict):
        try:
            batch_questions.dim()
        except AttributeError:
            batch_questions = torch.LongTensor(batch_questions)
        if batch_questions.dim() == 1:
            batch_questions = batch_questions.unsqueeze(dim = 0)
        # batch_contexts:  ([0, 1, 3, 4, 6, 7, 9, 10, 12, 13], [0, 1, 3, 4, 6, 7], [15, 16, 18, 19, 21, 22], [15, 16, 18, 19, 21, 22, 24, 25], [0, 1, 3, 4, 6, 7, 9, 10])
        contexts = self.input_module(self.embed, batch_contexts, tree_dict)
        questions = self.question_module(self.embed, batch_questions)
        M = questions
        for i in range(self.update_times):
            M = self.memory(contexts, questions, M)
        prediction = self.answer(questions, M)
        return prediction
    
    def predict(self, output, batch_questions, tree_dict):
        #output = self.forward(batch_contexts, batch_questions, tree_dict)
        pred = F.softmax(output, dim = 1)
        _, idx = torch.max(pred, dim = 1)
        return idx
            
def metrics(pred, target):
    try:
        target.dim()
    except AttributeError:
        target = torch.LongTensor(target)
    if pred.dim() < target.dim():
        target = target.view(-1)
    corrects = pred == target
    acc = torch.mean(corrects.float())
    return acc

            
class Control:
    LOG_PATH = 'log.txt'
    
    def __init__(self,
                 run,
                 task_id,
                 batch_size = 50,
                 embed_dim = 80,
                 mem_dim = 80,
                 sparse = True,
                 input_dropout = 0.1,
                 output_dropout = 0.1,
                 update_times = 3,
                 weight_decay = 0.001
                 ):
        self.run = run
        self.task_id = task_id
        self.batch_size = batch_size
        # 若 解析好的 Babidataset 存在就直接加载
        if os.path.exists(Babidataset.SAVE_PATH.format(task_id)):
            with open(Babidataset.SAVE_PATH.format(task_id), 'rb') as f:
                self.dataset = pickle.load(f, encoding = 'utf-8')
        else:
            self.dataset = Babidataset(task_id)
        vocab_size = len(self.dataset.vocab) 
        self.model = DMN(vocab_size = vocab_size,
                         embed_dim = embed_dim, 
                         mem_dim = mem_dim,
                         sparse = sparse,
                         input_dropout = input_dropout,
                         output_dropout = output_dropout,
                         update_times = update_times)        
        for name, param in self.model.state_dict().items():
            if ('weight' in name) and ('embed' not in name):
                init.xavier_normal_(param)
        self.model.cuda()
        self.criteria = nn.NLLLoss(size_average = False)
        self.metrics = metrics
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          weight_decay = weight_decay
                                          )
        self.early_stopping_cnt = 0
        self.early_stopping_flag = False
        self.best_acc = 0
        
        
    def train(self, epoch):
        self.dataset.mode = 'train'
        # 每一次取出 batch  对 batch 再按照contexts长度进行分组
        dataloader = DataLoad(self.dataset, batch_size = self.batch_size)
        self.model.train()
        total_acc = 0
        total_loss = 0
        cnt = 0
        # 0-6 7-12 13-18... 有多少组循环多少次  每个data里的context长度相似
        for batch_idx, data in enumerate(dataloader):
            self.optimizer.zero_grad()
            contexts, questions, answers = data
            batch_size = len(answers)
            '''
            batch_idx 0
            contexts ([0, 1, 3, 4, 6, 7, 9, 10, 12, 13], [0, 1, 3, 4, 6, 7], [15, 16, 18, 19, 21, 22], [15, 16, 18, 19, 21, 22, 24, 25], [0, 1, 3, 4, 6, 7, 9, 10])
            questions tensor([[ 10,  11,  14],
                    [ 10,  11,  12],
                    [ 10,  11,  14],
                    [ 10,  11,   7],
                    [ 10,  11,  12]])
            answers tensor([  5,   9,   5,  16,  16])
            '''
            output = self.model(contexts, questions, self.dataset.train_tree_dict)
            predict = self.model.predict(output, questions, self.dataset.train_tree_dict)
            m = nn.LogSoftmax(dim = 1)
            loss = self.criteria(m(output), answers)
            total_loss += loss.item()
            acc = self.metrics(predict, answers)
            total_acc += acc * batch_size
            cnt += batch_size 
            loss.backward()
            self.optimizer.step()
        with open(self.LOG_PATH, 'a+') as fp:
            fp.write(f'[Run {self.run}, Task {self.task_id}, Epoch {epoch}] [Training] loss : {total_loss / cnt: {10}.{8}}, acc : {total_acc / cnt: {5}.{4}}, batch_idx : {batch_idx}' + '\n')
        
    
    def valid(self, epoch):
        self.dataset.mode = 'valid'
        dataloader = DataLoad(self.dataset, batch_size = self.batch_size, shuffle = False)
        self.model.eval()
        total_acc = 0
        total_loss = 0
        cnt = 0
        for batch_idx, data in enumerate(dataloader):
            contexts, questions, answers = data
            batch_size = len(answers)
            output = self.model(contexts, questions, self.dataset.train_tree_dict)
            predict = self.model.predict(output, questions, self.dataset.train_tree_dict)
            acc = self.metrics(predict, answers)
            m = nn.LogSoftmax(dim = 1)
            loss = self.criteria(m(output), answers)
            print('|', batch_size, 'acc:', acc, 'loss:', loss,'|')
            total_loss += loss.item()
            total_acc += acc * batch_size
            cnt += batch_size 
        total_acc = total_acc / cnt
        if total_acc > self.best_acc:
            self.best_acc = total_acc
            self.best_state = self.model.state_dict()
            self.early_stopping_cnt = 0
        else:
            self.early_stopping_cnt += 1
            if self.early_stopping_cnt >= 256:
                self.early_stopping_flag = True
        with open(self.LOG_PATH, 'a+') as fp:
            fp.write(f'[Run {self.run}, Task {self.task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}} Loss : {total_loss / cnt: {5}.{4}}' + '\n')
        return total_acc
        
    def test(self):
        self.dataset.mode = 'test'
        dataloader = DataLoad(self.dataset, batch_size = self.batch_size, shuffle = False)
        self.model.load_state_dict(self.best_state)
        self.model.eval()
        total_acc = 0
        cnt = 0
        
        for batch_idx, data in enumerate(dataloader):
            contexts, questions, answers = data
            batch_size = len(answers)
            output = self.model(contexts, questions, self.dataset.test_tree_dict)
            predict = self.model.predict(output, questions, self.dataset.test_tree_dict)
            acc = self.metrics(predict, answers)
            total_acc += acc * batch_size
            cnt += batch_size
        with open(self.LOG_PATH, 'a+') as fp:
            fp.write(f'[Run {self.run}, Task {self.task_id}, Epoch {epoch}] [Test] Accuracy : {total_acc / cnt: {5}.{4}}'
                     + '\n')
        os.makedirs('models', exist_ok = True)
        with open(f'models/task{self.task_id}_epoch{epoch}_run{self.run}_acc{total_acc/cnt}.pth', 'wb') as fp:
            torch.save(self.model.state_dict(), fp)
            

if __name__ == '__main__':
    for run in range(constant.num_runs):
        for task_id in range(3,4):
            model = Control(run,
                            task_id,
                            batch_size = constant.batch_size,
                            embed_dim = constant.embed_dim,
                            mem_dim = constant.mem_dim,
                            sparse = constant.sparse,
                            input_dropout = constant.input_dropout,
                            output_dropout = constant.output_dropout,
                            update_times = constant.update_times,
                            weight_decay = constant.weight_decay
                            )
            for epoch in range(constant.num_epoch):
                if not model.early_stopping_flag:
                    model.train(epoch)
                    model.valid(epoch)
                else:
                    with open(model.LOG_PATH, 'a+') as fp:
                        fp.write(f'[Run {run}, Task {task_id}] Early Stopping at Epoch {epoch}, Valid Accuracy : {model.best_acc: {5}.{4}}' + '\n')
                    break
            model.test()