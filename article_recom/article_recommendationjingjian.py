
# coding: utf-8

# 实现功能：
# 采用两个数据集进行文章相似度推荐，第一个数据集为搜狗数据集，第二个数据集为本次业务数据集。数据集位于 dataset文件夹，因为系统问题上传失败。  
# 最后只需要运行 fun1, func2函数，func1实现对原始数据进行预处理，把处理后的中间文件存放在本地，服务器对该函数只需要间隔几个小时运行一次即可。func2读取func1的结果，对新传入的文章进行相似度比较，返回相似度较最高的几篇文章推荐，推荐的文章位于func1的返回文件中。

# ### 1. 加载package

# In[1]:


import glob
import re
import jieba
import jieba.posseg as pseg
import jieba.analyse
import os
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent import futures
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import dill
import pickle
import io

np.random.seed(42)
random.seed(42)

MAX_WORKERS = 20

# 预先定义停用词路径
stop_filepath = "./chinese_stop_words.txt"


# ### 2. 预定义函数

# In[26]:



def load_stopwords(stop_filepath):
    '''
    根据停用词文件所在的路径加载停用词，返回停用词列表
    
    参数
    ----------
    stop_filepath : string
        停用词文件所在的路径
        
    返回
    stopwords : list
        储存停用词
    ----------
    
    '''
#     stop_filepath = "/Users/cyy7645/Documents/internship/article_recom/chinese_stop_words.txt"
    stopwords = [line.strip() for line in 
                 io.open(stop_filepath, 'r', encoding='utf-8').readlines()]
    
    return stopwords


def participle(article, cut_all = False, cut_for_search = False):
    '''
    对文章进行分词，可选择分词方法
    
    参数
    ----------
    article : string
        代表整片文章的字符串
    cut_all : bool 默认 False
        当值为True时，使用全模式分词
    cut_for_search : bool 默认 False
        当值为True时，使用搜索引擎模式
        
    返回
    ----------
    article : string
        分完词的文章，词和词间隔为空格
    '''
    if jieba.cut_for_search == True:
        # 搜索引擎模式
        article = ' '.join(jieba.cut_for_search(article))
    elif cut_all == False:
        # 使用精准模式分词
        article = ' '.join(jieba.cut(article))
    else:
        # 使用全模式分词
        article = ' '.join(jieba.cut(article, cut_all=True))
        
    return article


def clear_stopwords(article, stopwords):
    '''
    对分词完成的每篇文章去除停用词
    
    参数
    ----------
    article : string
        分词后的文章，词和词间隔为空格
    stopwords : list
        储存停用词的列表
    
    返回
    ----------
    article : string
        删除通用词后的文章，词和词间隔为空格
    '''
    article = ' '.join(word for word in article.split() if word not in stopwords)
    
    return article

        
def divide_words(contexts,stopwords, cut_all = False, cut_for_search = False):
    '''
    对文章整个文章集(包含某一类的所有文章)进行分词
    
    参数
    ----------
    contexts : list
        包含整个文章集的列表，每个元素是一篇文章
    stopwords : list
        包含停用词的列表
    cut_all : bool 默认 False
        当值为True时，使用全模式分词
    cut_for_search : bool 默认 False
        当值为True时，使用搜索引擎模式
        
    返回
    ----------
    articles_list : list
        包含所有唯一文章的列表，其中的文章已经分词，去除停用词
        
    '''
#     # 每篇文章都是以 (责任编辑:马莹莹) 结尾，基于这个原理提取出文章存入list
#     articles_list = articles_in_category.strip().split(')\n')
    # 对list中的文章去重
    for i in range(len(contexts)):
        contexts[i] = participle(contexts[i], cut_all = False,                                       cut_for_search = False)
        contexts[i] = clear_stopwords(contexts[i],stopwords)
        
    # 返回经过分词 去除停用词后的文章集
    return contexts




# 从每篇文章中提取关键词
def extract_keywords(article, topK=20, withWeight=True, allowPOS=()):
    '''
    根据单词在文章中出现的频率从每篇文章中提取关键词，返回一个包含关键词的列表和包含主体对应权重的列表
    
    参数
    ----------
    article: string
        表示文章的字符串
    topK : int 默认 20
        返回几个 TF/IDF 权重最大的关键词
    withWeight : bool 默认 True    
        是否一并返回关键词权重值
    allowPOS : tuple 默认为空
        仅包括指定词性的词
        
    返回
    ----------
    items : list 
        包含关键词的列表
    weights : list
        关键词对应的权重
    '''
    keywords = jieba.analyse.extract_tags(article, topK=20, withWeight=True, allowPOS=())
    items = []
    weights = []
    for item in keywords:
    # 分别为关键词和相应的权重
        items.append(item[0])
        weights.append(item[1])
        
    return items, weights


def split_vectors_and_classes(articles_str, articles_classes):
    '''
    生成TF-IDF表，并把储存文章和类别的列表划分训练集和测试集
    
    参数
    ----------
    articles_str : list
        打乱顺序后整个数据集的文章(关键词)
        articles_str可能是一维数组（文章），也可能是二维数组（关键词）
    articles_classes : list
        打乱顺序后整个数据集的文章所对应的类[int]
        
    返回
    ----------
    vectors : list
        IF-IDF表
    train_vectors : list
        训练集的IF-IDF表
    test_vectors : list
        测试集的IF-IDF表
    articles_classes_test_np : numpy
        测试集文章所对应的类别
    articles_classes_train_np : numpy
        训练集文章所对应的类别
    vectorizer : class
        文本特征提取器    
    '''
    vectorizer = TfidfVectorizer()
    if isinstance(articles_str[0], list) == False:
        vectors = vectorizer.fit_transform(articles_str)
    else:
        articles_str = [' '.join(x) for x in articles_str]
        vectors = vectorizer.fit_transform(articles_str)
    train_vectors = vectors[: int(9 * vectors.shape[0]/10), :]
    test_vectors = vectors[int(-vectors.shape[0]/10): , :]
    articles_classes_test = articles_classes[int(-len(articles_classes)/10):]
    articles_classes_train = articles_classes[:int(9 * len(articles_classes)/10)]
    articles_classes_test_np = np.array(articles_classes_test)
    articles_classes_train_np = np.array(articles_classes_train)
    
    # 返回训练/测试TF-IDF矩阵  训练/测试类别(numpy类型) vectorizer
    return train_vectors, test_vectors, articles_classes_test_np,             articles_classes_train_np, vectorizer, vectors


def get_contents_from_csv(path, days):
    '''
    从.csv文件中提取文章的id、创建日期、标题、描述和内容
    
    参数
    ----------
    path : string
        .csv文件所在的路径
    
    返回
    ----------
    id : list
        存放文章对应的id
    times : list
        存放某一类别下文章的创建日期 只保留年月日
    titles : list
        存放某一类别下所有文章的标题
    descriptions : list
        存放某一类别下所有文章的描述
    contents : list
        存放某一类别下所有文章的内容
    '''
    fields = ['id','create_time','title', 'description','content']
    df = pd.read_csv(path, usecols=fields)
    
    df['create_time'] = pd.to_datetime(df['create_time'])
    latest_date = df['create_time'].max()
    earlist_date = latest_date-np.timedelta64(days,'D')
    df = df[(df['create_time'] > earlist_date) & (df['create_time'] <= latest_date)]
    
    df = df.drop_duplicates(subset=['description'], keep=False)
    ids = df['id'].tolist()
    times = df['create_time'].tolist()
#     times = [x.split()[0] for x in times]
    titles = df['title'].tolist()
    descriptions = df['description'].tolist()
    original_contents = df['content'].tolist()
    # 根据文件的内容储存方式使用正则表达式提取内容
    symbol = re.compile(r'\<.*?\>')
    contents = []
    for content in original_contents:
        cleared_content = symbol.sub('',content)
        cleared_content = ''.join(cleared_content.split('\n'))
        contents.append(cleared_content)
    return ids, times, titles, descriptions, contents

def combine_title_desc_content(titles, descriptions, contents):
    '''
    把文章的标题、描述和内容合并成一个列表，并保证一一对应
    
    参数
    ----------
    titles : list
        存放某一类别下所有文章的标题
    descriptions : list
        存放某一类别下所有文章的描述
    contents : list
        存放某一类别下所有文章的内容
        
    返回
    ----------
    combined : list
    文章的标题、描述和内容合并后的列表
    '''
    combined = [m+str(n)+k for m,n,k in zip(titles,descriptions,contents)]
    
    return combined


# ### 4. 对新文章进行预处理函数

#  得到一篇新的文章，在其所属的类中根据相似度指标找出相似度最高的一篇

# In[30]:


def preprocess_data_to_keywords(path, stopwords, days):
    '''
    对原始数据集进行 分词、去除停用词、去重，
    然后生成储存文章关键字和其所属类别的 list和dict
    
    参数
    ----------
    path : string
        预处理完成的数据所在的路径，e.g. path = 
                    '/Users/cyy7645/Documents/internship/data/csv_news/*.csv'
    days : int 默认1000
        从今天开始往前n天的数据作为相似度比较的数据
    
    返回
    ----------
    items_str : list
        储存整个数据集的文章对应的关键字
    items_classes : list
        储存整个数据集的文章对应的关键字所对应的类[int]
    ids_items : dict
        储存文章对应的关键字和其类别一一对应的字典
        value为文章对应的关键字，key为文章对应的类 int
    ids_classes : dict
        储存文章id和其所属类别的字典
        key为文章的id，value为该文章所属的类别
    label_classes : dict
        储存标签和id的字典
        key为标签，value为id
    
    '''
    divided_paths = glob.glob(path)
    items_str = []
    ids_items = {}
    items_classes = []
    ids_classes = {}
    label_classes = {}
    count_articles = 1
    article_class = 1
    for path in divided_paths:
        print("start process ",path.split('/')[-1],"...")
        ids, times, titles, descriptions, contents = get_contents_from_csv(path, days)
        combined = combine_title_desc_content(titles, descriptions, contents)
        combined = divide_words(combined,stopwords)
        df = pd.DataFrame({'id':ids,'create_time': times,'context': combined})
        df = df[['id','create_time', 'context']]
        

        print(df.shape)
        articles_dict = {}
        label = path.split('/')[-1].split('.')[0]
        for index, row in df.iterrows():
            context = row['context']
            id_num = row['id']
            items,weights = extract_keywords(context, topK=30, 
                                             withWeight=True, allowPOS=())
            items_str.append(items)
            ids_items[id_num] = items
            items_classes.append(article_class)
            ids_classes[id_num] = article_class
        label_classes[label] = article_class
        article_class += 1
        print("preprocess ",path.split('/')[-1],"done ...")
    return items_str, items_classes, ids_items, ids_classes, label_classes


# In[71]:


def preprocess_new_article(doc_path, stop_filepath, vectorizer, items_classes, vectors,items_str, ids_items,
                           cut_all = False, cut_for_search = False,  n_docs = 3, label = None):
    '''
    对新输入的文章进行预处理：分词，去除停用词，提取关键词，得到TF-IDF值和数据库中的所有文章进行相似度比较，
                          返回相似度最高的n篇文章，并把新文章写入到一个文件（因为对此后新输入的文章需要同
                          此篇文章进行相似度比较）
    
    参数
    ----------
    path : string
        新文章所在的路径
    stop_filepath : string
        停用词文件所在的路径
    cut_all : bool 默认 False
        当值为True时，使用全模式分词
    cut_for_search : bool 默认 False
        当值为True时，使用搜索引擎模式
    vectorizer: class
        文本特征提取器
    clf : object
        分类器
    n_docs : int 默认为3
        返回几篇最相近的文章
    label : string 默认 None
        新文章所属的类别
    
    返回
    ----------
    most_simi_doc : list
        与输入文章最相似的文章列表
    '''
    keys = []
    with open (doc_path, 'r') as p:
        new_doc = p.read().split("\n")
        new_doc = ' '.join(new_doc)
    keywords,weights = extract_keywords(new_doc, topK=30, withWeight=True, allowPOS=())
    new_doc = ' '.join(keywords)
    new_doc = [new_doc]
    
 
    # 得到这篇文章的tf-idf值
    vectors_new_doc = vectorizer.transform(new_doc)
    
    
    # 计算新文章与数据库中文章的相似度
    cosine_similarities = linear_kernel(vectors[:], vectors_new_doc).flatten()
    
    # 根据相关性找到最相似的文章编号
    related_docs_indices = cosine_similarities.argsort()[:-100:-1]
#     print(related_docs_indices)
#     print(sorted(cosine_similarities.tolist())[-5:])
    most_simi_doc = []
    while n_docs > 0:
        simi_doc = ' '.join(items_str[int(related_docs_indices[-n_docs])])
        most_simi_doc.append(simi_doc)
        keys.append(list(ids_items.keys())[list(ids_items.values()).index(items_str[int(related_docs_indices[-n_docs])])])
        n_docs -= 1
    reversed(most_simi_doc)
    
#     dicxx = {'a':'001', 'b':'002'}
#     list(ids_items.keys())[list(ids_items.values()).index("001")]
    
#     # 把这篇新文章的关键词写入到文件中
#     with open("/Users/cyy7645/Desktop/new_docs.txt","r+") as f:
#         f.read()
#         f.write('\n')
#         f.write(new_doc)
        
    return most_simi_doc, keys


# ### 5. 储存加载中间变量，方便func2从本地调用

# In[32]:


def store_tmp(store_path,vectorizer, vectors, label_classes, items_classes, items_str,ids_items):
    '''
    储存func1中间文件的函数
    '''
    with open(store_path + 'vectorizer', 'wb') as f:
        dill.dump(vectorizer, f)
    with open(store_path + 'vectors', 'wb') as f:
        dill.dump(vectors, f)
    with open(store_path + 'label_classes', 'wb') as f:
        dill.dump(label_classes, f)
    with open(store_path + 'items_classes', 'wb') as f:
        dill.dump(items_classes, f)
    with open(store_path + 'items_str', 'wb') as f:
        dill.dump(items_str, f)
    with open(store_path + 'ids_items', 'wb') as f:
        dill.dump(ids_items, f)
        
def restore_tmp(restore_path):
    '''
    加载func1中间文件的函数
    '''
    with open(store_path + 'vectorizer', 'rb') as f:
        vectorizer = dill.load(f)
    with open(store_path + 'vectors', 'rb') as f:
        vectors = dill.load(f)
    with open(store_path + 'label_classes', 'rb') as f:
        label_classes = dill.load(f)
    with open(store_path + 'items_classes', 'rb') as f:
        items_classes = dill.load(f)
    with open(store_path + 'items_str', 'rb') as f:
        items_str = dill.load(f)
    with open(store_path + 'ids_items', 'rb') as f:
        ids_items = dill.load(f)
    return vectorizer, vectors, label_classes, items_classes, items_str,ids_items


# ### 6. 对外接口函数   每隔一段时间运行func1, 为新输入文章找相似文章运行func2

# 以下两个函数是对上述函数的封装，作为对外提供的api

# In[37]:


# 函数一：每隔一段时间运行

def func1(path, stop_filepath, store_path, days):
    '''
    对原始数据集进行预处理，包括分词、去除停用词、生成TF-IDF表等，最后将结果存在本地，方便func2调用
    
    参数
    ----------------
    path : string
        原始数据集路径
    stop_filepath : string
        停用词词表的存放路径
    store_path : string
        存放结果的路径
        
    返回值
    ----------------
    无
    '''
    # 若 stop_filepath文件夹不存在，创建
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    # 加载停用词表
    stopwords = load_stopwords(stop_filepath)
    items_str, items_classes, ids_items, ids_classes, label_classes = preprocess_data_to_keywords(path, stopwords, days)
    train_vectors, test_vectors, articles_classes_test_np, articles_classes_train_np,     vectorizer, vectors =  split_vectors_and_classes(items_str, items_classes)
    store_tmp(store_path,vectorizer, vectors, label_classes, items_classes, items_str,ids_items)
    


# In[38]:


# 运行 func1 函数
store_path = './tmp/'
path = u"./news/*.csv"
func1(path,stop_filepath,store_path, days=10)


# In[75]:


def func2(path, stop_filepath, write_path,restore_path):
    '''
    读取func1的结果，根据新文章所在的地方，把相似文章的id写入到文件中
    
    参数
    ------------------
    path : string
        待比较相似度的文章所在的地址
    stop_filepath : string
        停用词词表的存放路径
    write_path : string
        相似文章id结果存放路径
    restore_path : string
        func1结果存放的路径
        
    返回值
    ------------------
    most_simi_doc : list
        相似度最高的文章列表
    keys : list
        相似度最高的文章对应id的列表
    '''
    
    vectorizer, vectors, label_classes, items_classes, items_str,ids_items = restore_tmp(restore_path)
    
    most_simi_doc, keys = preprocess_new_article(path, stop_filepath, vectorizer, items_classes, vectors, items_str,ids_items)
    f = open(write_path, 'w')
    for i in range(len(keys)):
        if i < len(keys)-1:
            f.write(str(keys[i]))
            f.write(',')
        else:
            f.write(str(keys[i]))
    f.close()
    return most_simi_doc,keys


# In[76]:


# 运行 func2 函数
restore_path = './tmp/'
doc_path = "./new_doc.txt"
write_path = "./write_keys.txt"
most_simi_doc,keys  = func2(doc_path, stop_filepath, write_path,restore_path)


# In[77]:


print(most_simi_doc)
print(keys)

