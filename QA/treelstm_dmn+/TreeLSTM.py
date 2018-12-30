import torch
import torch.nn as nn
from SentenceTreeLSTM import SentenceTreeLSTM

# 对contexts中的所有句子使用TreeLSTM，返回高维tensor 代表所有句子的根结点隐藏层
class TreeLSTM(nn.Module):
    #                  241 词向量维度：300      150      50           5        false     false
    def __init__(self, in_dim,mem_dim, hidden_dim, num_classes):
        super(TreeLSTM, self).__init__()
        # 定义embedding层
        self.sentencetreelstm = SentenceTreeLSTM(in_dim, mem_dim)
        # self.similarity = Similarity(mem_dim, hidden_dim, num_classes)
    # ctree, qtree, csent, qsent
    def forward(self, cid,  csent,word_embedding,tree_dict):
        # 把 linputs，rinputs做embedding  假设7个单词，7*300  linputs：数字

        # 把ctree和tree_dict合并成 一个tree
        trees = []
        for id in cid:
            root = tree_dict[int(id)]
            trees.append(root)
            # trees中存储 一个contexts中的句子的root

        cinputss = []
        for i in range(len(csent)):
            cinputs = word_embedding(torch.tensor(csent[i], dtype=torch.long, device='cpu').cuda())
            cinputs = torch.unsqueeze(cinputs, 1)
            cinputss.append(cinputs)

        cstates = []
        for sen in range(len(cinputss)):
            cstate, chidden = self.sentencetreelstm(trees[sen], cinputss[sen])
            cstates.append(cstate)
        cstates = torch.stack(cstates)
        cstates = torch.squeeze(cstates)

        return cstates