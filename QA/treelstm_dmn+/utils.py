from vocab import Vocab
import os
import math

import torch

# loading GLOVE word vectors  加载训练好的词向量
# if .pth file is found, will load that
# else will load from .txt file & save
# path = '../data/gloveglove.840B.300d.path'
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        # 加载权重库和字典里词向量权重
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    else:
        print("need to be handled in utils.py")