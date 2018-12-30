import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用tree-lstm计算每棵树 根结点的隐藏层
class SentenceTreeLSTM(nn.Module):
    # in_dim:词向量维度 300   mem_dim: 隐藏层维度 150
    def __init__(self, in_dim, mem_dim):
        super(SentenceTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        # 定义线性方程
        #                           300          450
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        #                           150          450
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)


    # 通过tree-LSTM 得到输出
    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        # print(inputs.size())
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        # 返回 最后时刻的memory cell 和 hidden state  1*150
        return c, h
############################################################
    # tree对象 和 词向量 7*300  把tree换成二维tensor 通过BFS得到
    def forward(self, tree, inputs):
        # leaves, inodes = self.BFStree(tree)
        # print("leaves:",leaves)
        # print("inodes:", inodes)
        # 递归 找到叶子结点 the
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            # 得到the这个叶子结点对应的 memory cell和hidden state 1 * 150
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            # (tensor([[-0.1957, -0.0413..150]]),)  把多个叶子结点的cell和hidden合并
            #  The * in a function call "unpacks" a list (or other iterable),
            # making each of its elements a separate argument. zip([[1,2,3],[4,5,6]]) --> zip([1,2,3], [4,5,6])
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            # tensor([[-0.1957, -0.0413..150]])
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            # print("child_c:",child_c)
            # print("child_h:", child_h)
        # inputs[tree.idx]根据树的编号找到单词对应的词向量    1*150的tensor     返回(1*150的tensor,1*150的tensor)  代表memory cell和hidden state
        # print(inputs)
        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)

        # 通过递归 返回树的cell 和 hidden state
        return tree.state
