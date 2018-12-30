# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 12:29:39 2018

@author: 陈俏均
"""

#tree class
import os
from nltk.parse.stanford import StanfordDependencyParser
import re 

#java_path = 'C:/Program Files/Java/jdk1.8.0_181/bin/java.exe'
#os.environ['JAVAHOME'] = java_path
os.environ['STANFORD_PARSER'] = '/data/notebook/jupyterhub/notebook_dirs/chenyy/QA/treelstm_dmn+/lib/stanford-parser/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/data/notebook/jupyterhub/notebook_dirs/chenyy/QA/treelstm_dmn+/lib/stanford-parser/stanford-parser-3.9.1-models.jar'

# 定义树结构
class Tree:
       
    def __init__(self):
        # 类中包含的属性
        self.idx = None
        self.parent = None
        self.child = []
        self.num_child = 0
        self._depth = 0
        self._size = 1
    
    @property 
    def depth(self):
        return self._depth    
        
    @depth.setter
    def depth(self, value):
        if self.depth <= value:
            self._depth = value + 1
        if self.parent is not None:
            self.parent.depth = self.depth
    
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size += value
        if self.parent is not None:
            self.parent.size = value

    # 为当前树增加孩子
    def add_child(self, child):
        child.parent = self
        self.child.append(child)
        self.num_child += 1
        self.depth = child.depth
        self.size = child.size

# 使用解析包把句子解析成依存树，返回值result，为一维list，包含句子中单词的父子关系
def get_line_func():
    parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    def get_line(context):
        try:
            context.split()
        except AttributeError:
            pass
        else:
            context = context.split()

        result, = parser.parse(context)
        res = result.to_conll(4)
        nums = re.findall(r'\d+\.?\d*', res)
        results = list(map(int, nums))
        return results
    return get_line


# 根据解析树产生的变量line(包含句子中单词的父子关系)，构建出树结构
# e.g line: [2, 0, 2, 5, 2, 2]
#     内容: "Mary picked up the apple there" 
# 0 代表根结点,line中的id和内容中的单词一一对应，遍历line中的每个元素，找到对应单词和其他单词间的父子关系
# 比如对2 而言，对应单词Mary的父亲是 picked，picked对应的id为0，因此picked是根结点
def read_tree(line, context):
    parents = line
    trees = dict()
    root = None
    for i in range(1, len(parents) + 1):
        if i - 1 not in trees.keys() and parents[i - 1] != -1:
            idx = i
            prev = None
            while True:
                parent = parents[idx - 1]
                if parent == -1:
                    break
                tree = Tree()
                if prev is not None:
                    tree.add_child(prev)
                trees[idx - 1] = tree
                tree.idx = idx - 1
                tree.idx = context[tree.idx]
                if parent - 1 in trees.keys():
                    trees[parent - 1].add_child(tree)
                    break
                elif parent == 0:
                    root = tree
                    break
                else:
                    prev = tree
                    idx = parent
    return root

get_line = get_line_func()