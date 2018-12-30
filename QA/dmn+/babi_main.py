# -*- coding:utf-8 -*-
from babi_loader import BabiDataset, pad_collate
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader


# input module 的positional encoder
def position_encoding(embedded_sentence):
    '''
    假设batch为2，sentence为4，token为7，enmedding为80
    embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
    l.size() -> (#sentence, #embedding)
    output.size() -> (#batch, #sentence, #embedding)
    '''
    # 因为需要tensor运算  所以需要padding
    _, _, slen, elen = embedded_sentence.size()
    # 假设有个批次中左右的context有7个句子， 词向量维度为80
    # l为二维list，代表词向量的权重，7*80 
    l = [[(1 - s / (slen - 1)) - (e / (elen - 1)) * (1 - 2 * s / (slen - 1)) for e in range(elen)] for s in range(slen)]
    # 把list转变为tensor
    l = torch.FloatTensor(l)
    # 返回一个新的张量，对输入的指定位置插入维度 1
    l = l.unsqueeze(0)  # for #batch
    l = l.unsqueeze(1)  # for #sen
    # 1*1*7*80

    # 将l进行扩充,恢复成2*4*7*80,其中2*4是对7*80的重复
    l = l.expand_as(embedded_sentence)
    # embedded_sentence和l点乘 size=2*4*7*80
    weighted = embedded_sentence * Variable(l.cuda())
    # 把第三维的数相加，变为2*4*80
    return torch.sum(weighted, dim=2).squeeze(2)  # sum with tokens

# 时间记忆模块attention mechanism 的 Attention based GRU
class AttentionGRUCell(nn.Module):
    # 80   80
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        # 80
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.U.state_dict()['weight'])

    # GRU结构，对DMN的改进
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
                # C: 1*2*80 --> 2*80
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
        init.xavier_normal(self.z1.state_dict()['weight'])
        init.xavier_normal(self.z2.state_dict()['weight'])
        init.xavier_normal(self.next_mem.state_dict()['weight'])

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
        print("G.size():",G.size())
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
        # G: 2*8
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
        print("next_mem.size():",next_mem.size())
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
        print("questions.size():",questions.size())
        # 交换0,1维的值
        questions = questions.transpose(0, 1)
        return questions


# 接受contexts
class InputModule(nn.Module):
    # 该类继承于nn.Modul，重新定义其构造函数
    def __init__(self, vocab_size, hidden_size):
        # super(InputModule,self) 首先找到 InputModule 的父类（也就是类 nn.Module），
        # 然后把类的对象 InputModule 转换为类 nn.Module 的对象
        # super()函数来调用父类（nn.Module）的init()函数，解决对冲继承问题
        super(InputModule, self).__init__()
        # 除了父类构造函数中的成员变量和成员函数，额外创建新的成员变量和成员函数

        # hidden_size=80
        self.hidden_size = hidden_size
        # 双向gru,输入的特征值维度,GRU看做多层神经网络，有多个weight，如layer1.weight layer2.weight
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        # name是成员变量或函数名，param值的是变量本身或函数本身
        # 找到所有的weight，对weight进行初始化
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    # 把input转化为词向量，再把词向量通过双向GRU获得fi，Input Fusion Layer
    def forward(self, contexts, word_embedding):
        '''
        contexts.size() -> (#batch, #sentence, #token)
        word_embedding() -> (#batch, #sentence, #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        '''
        # 2，4，7
        batch_num, sen_num, token_num = contexts.size()

        # 改变context的尺寸，变成 #batch * （#sentence * #token）
        contexts = contexts.view(batch_num, -1)
        # 把词转换成词向量 2*28   2*28*80
        contexts = word_embedding(contexts)
        # 把context的尺寸改回来，-1代表 词向量维数（80） 2*4*7*80
        contexts = contexts.view(batch_num, sen_num, token_num, -1)
        # 维度变成2*4*80 batch_size * sen_num * embedding
        contexts = position_encoding(contexts)
        # 把10%的值变成0
        contexts = self.dropout(contexts)

        # h0维度 2*2*80, 因为bidirectional所以第一维是2
        h0 = Variable(torch.zeros(2, batch_num, self.hidden_size).cuda())
        # 通过gru对context,h0更新，# 拼接 2, 4, 160 + 2, 2, 80
        facts, hdn = self.gru(contexts, h0)
        print(facts.size(), hdn.size())
        # 把facts分割成两部分，为两部分的对应元素相加
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        print("facts.size():", facts.size())
        return facts

class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)
    # M: 42  questions:80
    # 通过q和m获得answer
    def forward(self, M, questions):
        M = self.dropout(M)
        # M: 2*1*80
        # questions: 2*1*80
        print("M.size():", M.size())
        print("questions.size():", questions.size())
        concat = torch.cat([M, questions], dim=2).squeeze(1)
        z = self.z(concat)
        print("z.size():", z.size())
        return z


# 继承于nn.Module，构建模型结构，包含input,question,memory,answer模块
class DMNPlus(nn.Module):
    # hidden_size=80 词向量大小, vocab_size=41 字典大小, num_hop=3 回答模块迭代次数, qa=dset.QA 语料库字典
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        # 函数是用于调用父类(超类)DMNPlus
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        # sparse=True通常运行速度更快
        # 根据字典大小和嵌入向量大小，把每一个输入转换为嵌入向量大小的tensor
        # 当输入context 4*7时，输出 4*7*80
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True).cuda()

        # 从均匀分布U(-sqrt(3), sqrt(3))中生成值，填充输入的张量或变量
        # state_dict()是一个字典，存放如weight，所以这一步是初始化weight
        # state_dict (dict) – 保存parameters和persistent buffers的字典。
        # 保存着module的所有状态（state）
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3 ** 0.5), b=3 ** 0.5)
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
        facts = self.input_module(contexts, self.word_embedding)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        # 更新三次memory
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)
        print("M.size():",M.size())
        preds = self.answer_module(M, questions)
        # preds: 2*160
        return preds


    def get_loss(self, contexts, questions, targets):
        output = self.forward(contexts, questions)
        # target 为answer对应单词的id
        print("targets.size():",targets.size())
        print("output.size():", output.size())
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
    # 训练十次
    for run in range(10):
        # 20个训练任务
        for task_id in range(1, 21):
            # dset是BabiDataset类的一个实例
            dset = BabiDataset(task_id)
            # 语料库的个数
            vocab_size = len(dset.QA.VOCAB)
            # 隐藏层结点数
            hidden_size = 80
            # 实例化预定义的模型
            model = DMNPlus(hidden_size, vocab_size, num_hop=3, qa=dset.QA)
            # 把模型放进GPU
            model.cuda()
            # 记录当前accuracy < best的次数
            early_stopping_cnt = 0
            # 停止的标记
            early_stopping_flag = False
            best_acc = 0
            # 定义优化器
            optim = torch.optim.Adam(model.parameters())

            # epoch=256
            for epoch in range(256):
                dset.set_mode('train')
                # 每次迭代用100个batch
                train_loader = DataLoader(
                    dset, batch_size=2, shuffle=True, collate_fn=pad_collate
                )

                model.train()
                # early_stopping为设定时
                if not early_stopping_flag:
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(train_loader):
                        optim.zero_grad()
                        contexts, questions, answers = data
                        batch_size = contexts.size()[0]
                        # 把内容，问题，答案放进GPU
                        contexts = Variable(contexts.long().cuda())
                        questions = Variable(questions.long()).cuda()
                        answers = Variable(answers).cuda()

                        # 获得 loss和acc
                        loss, acc = model.get_loss(contexts, questions, answers)
                        # 反向传播
                        loss.backward()
                        # acc*100作为总acc
                        total_acc += acc * batch_size
                        # 记录batch_size
                        cnt += batch_size

                        # 如果连续20个epoch accuracy都没都没有提升就停止
                        if batch_idx % 20 == 0:
                            print(f'[Task {task_id}, Epoch {epoch}] [Training] loss : {loss.data[0]: {10}.{8}}, acc : {total_acc / cnt: {5}.{4}}, batch_idx : {batch_idx}')
                        optim.step()
                    # 用validation验证结果
                    dset.set_mode('valid')
                    valid_loader = DataLoader(
                        dset, batch_size=2, shuffle=False, collate_fn=pad_collate
                    )

                    model.eval()
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(valid_loader):
                        contexts, questions, answers = data
                        batch_size = contexts.size()[0]
                        contexts = Variable(contexts.long().cuda())
                        questions = Variable(questions.long().cuda())
                        answers = Variable(answers).cuda()

                        _, acc = model.get_loss(contexts, questions, answers)
                        total_acc += acc * batch_size
                        cnt += batch_size
                    # 验证数据集 的平均准确率
                    total_acc = total_acc / cnt
                    if total_acc > best_acc:
                        best_acc = total_acc
                        best_state = model.state_dict()
                        early_stopping_cnt = 0

                    # 若验证集测试的准确率连续20次不再提升，early stop
                    else:
                        early_stopping_cnt += 1
                        if early_stopping_cnt > 20:
                            early_stopping_flag = True

                    print(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}}')
                    # 把验证集的准确率和loss写入日志文件
                    with open('log.txt', 'a') as fp:
                        fp.write(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}}' + '\n')
                    if total_acc == 1.0:
                        break
                else:
                    print(f'[Run {run}, Task {task_id}] Early Stopping at Epoch {epoch}, Valid Accuracy : {best_acc: {5}.{4}}')
                    break

            dset.set_mode('test')
            test_loader = DataLoader(
                dset, batch_size=2, shuffle=False, collate_fn=pad_collate
            )
            test_acc = 0
            cnt = 0

            for batch_idx, data in enumerate(test_loader):
                contexts, questions, answers = data
                batch_size = contexts.size()[0]
                contexts = Variable(contexts.long().cuda())
                questions = Variable(questions.long().cuda())
                answers = Variable(answers).cuda()

                model.load_state_dict(best_state)
                _, acc = model.get_loss(contexts, questions, answers)
                test_acc += acc * batch_size
                cnt += batch_size
            print(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Test] Accuracy : {test_acc / cnt: {5}.{4}}')
            # 储存模型到 models 文件夹
            os.makedirs('models', exist_ok=True)
            with open(f'models/task{task_id}_epoch{epoch}_run{run}_acc{test_acc/cnt}.pth', 'wb') as fp:
                torch.save(model.state_dict(), fp)
            with open('log.txt', 'a') as fp:
                fp.write(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Test] Accuracy : {total_acc: {5}.{4}}' + '\n')