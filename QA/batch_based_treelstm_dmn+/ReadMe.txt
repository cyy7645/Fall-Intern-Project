Implement batch based TreeLSTM model, which can run on GPU in batch.

run_on_gpu_1.ipynb, run_on_gpu_1.ipynb: includes very detailed introduction of all models, how it developed.

constant.py :Commands for running on GPU.

dataload.py : implementation of DataLoad class, form data into a batch.

parse_tree.py : data pre-processing, includes parsing sentence method, the parsed data is in parsed_dataset folder. The parsed data is different from treelstm_dmn+ model.

parser_log.txt : the log of running parse_tree.py, with current progress and estimated remaining time.

tree.py : definition the structure of tree.

parsed_data: parsed dataset.


torchfold.py: Core code includes the model, how to train and test, save log into log.txt. The input is parsed dataset in parsed_dataset folder.

log.txt : training log.
