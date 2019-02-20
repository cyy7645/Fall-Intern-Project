This folder includes pre-processing, classes for sparsing sentence and modeling, parsed data and original data of TreeLSTM_DMN+.

run_on_gpu_1.ipynb run_on_gpu_2.ipynb : Commands for running on GPU, check the using rate of GPU.

treelstm_dmn_babi_loader.py : data pre-processing, includes parsing sentence method, the parsed data is in parsed_dataset folder.

constants.py : definition of constant.

adict.py : class for dictionary structure.

Tree.py : class for tree structure.

SentenceTreeLSTM.py : apply iteration for tree(which is a sentence) with TreeLSTM, return the hiden state.

TreeLSTM.py : apply iteration for content(which is multiple sentences) with TreeLSTM, return the concatenated hidden state.

treelstm_dmn_babi_main.py : Core code includes the model, how to train and test, save log into log.txt. The input is parsed dataset in parsed_dataset folder.

log.txt : includes training and testing log for 20 tasks for 10 times (why 10 times? Eliminate errors caused by instability of model )

lib: jar files for parsing tree. Google stanford-parser.

glove: includes the library of Glove. It performs bad... skip it..

babi_main_glove.py : the only difference from treelstm_dmn_babi_main.py is that it apples GloVe.

utils.py : class for Glove related method, will be called in babi_main_glove.py.

log_for_every_task: traing and testing logg for every task, which is the seperated version of log.txt.


babi_loader_chinese.py 和 babi_main_chinese.py:
translate the dataset into Chinese, tried Chinese with the same model.


