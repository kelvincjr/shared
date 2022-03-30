import warnings
warnings.filterwarnings("ignore")

import os
import jieba
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
#sys.path.insert(0, "../typing_extensions-4.1.1/src")
#sys.path.insert(0, "../huggingface_hub-0.2.0/src")
#sys.path.insert(0, "../tokenizers-0.11.1/py_src/")
#sys.path.insert(0, "../transformers-4.17.0/src")
#sys.path.insert(0, '/data/kelvin/python/knowledge_graph/ai_contest/gaiic2022/baseline/ark-nlp-0.0.7')
#sys.path.insert(0, '/data/kelvin/python/knowledge_graph/ai_contest/gaiic2022/baseline/ark-nlp-main')
from ark_nlp.model.ner.global_pointer_bert import GlobalPointerBert
from ark_nlp.model.ner.global_pointer_bert import GlobalPointerBertConfig
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Task
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer

import os
from ark_nlp.factory.utils.conlleval import get_entity_bio

#data_path = '/data/kelvin/python/knowledge_graph/ai_contest/gaiic2022/baseline/baseline/data/'
data_path = './data/'
datalist = []
max_len = 0
len_count_32 = 0
len_count_64 = 0
with open(data_path + 'train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines.append('\n')
    
    text = []
    labels = []
    label_set = set()
    
    for line in lines: 
        if line == '\n':                
            text = ''.join(text)
            entity_labels = []
            for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                entity_labels.append({
                    'start_idx': _start_idx,
                    'end_idx': _end_idx,
                    'type': _type,
                    'entity': text[_start_idx: _end_idx+1]
                })
                
            if text == '':
                continue
            
            datalist.append({
                'text': text,
                'label': entity_labels
            })

            if len(text) > max_len:
                max_len = len(text)
            if len(text) > 32:
                len_count_32 += 1
            if len(text) > 64:
                len_count_64 += 1
            
            text = []
            labels = []
            
        elif line == '  O\n':
            text.append(' ')
            labels.append('O')
        else:
            line = line.strip('\n').split()
            if len(line) == 1:
                term = ' '
                label = line[0]
            else:
                term, label = line
            text.append(term)
            label_set.add(label.split('-')[-1])
            labels.append(label)
print('===== data preprocess done, datalist len: {}, len_count_32: {}, len_count_64: {}, max_len: {} ====='.format(len(datalist), len_count_32, len_count_64, max_len))

# 这里随意分割了一下看指标，建议实际使用sklearn分割或者交叉验证

train_data_df = pd.DataFrame(datalist[:-400])
train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

dev_data_df = pd.DataFrame(datalist[-400:])
dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))
print('===== dataframe init done =====')

ner_train_dataset = Dataset(train_data_df, categories=label_set)
ner_dev_dataset = Dataset(dev_data_df, categories=ner_train_dataset.categories)
print('===== dataset init done =====')

#tokenizer = Tokenizer(vocab='hfl/chinese-bert-wwm', max_seq_len=128)
#model_path = '/opt/kelvin/python/knowledge_graph/ai_contest/gaiic2022/baseline/model/bert_model'
model_path = 'hfl/chinese-bert-wwm'
tokenizer = Tokenizer(vocab=model_path, max_seq_len=128)
print('===== tokenizer init done =====')

ner_train_dataset.convert_to_ids(tokenizer)
print('===== train data convert_to_ids done =====')
ner_dev_dataset.convert_to_ids(tokenizer)
print('===== dev data convert_to_ids done =====')

config = GlobalPointerBertConfig.from_pretrained(model_path, num_labels=len(ner_train_dataset.cat2id))
torch.cuda.empty_cache()
dl_module = GlobalPointerBert.from_pretrained(model_path, config=config)
optimizer = get_default_model_optimizer(dl_module)
model = Task(dl_module, optimizer, 'gpce', cuda_device=0)


# 设置运行次数
num_epoches = 5
batch_size = 16

print('===== start to train =====')
model.fit(ner_train_dataset, 
          ner_dev_dataset,
          lr=2e-5,
          epochs=num_epoches, 
          batch_size=batch_size
         )

torch.save(model.module.state_dict(), './model_save.pth')