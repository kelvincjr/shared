#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : test
# @Author   : 研哥哥
# @Time     : 2020/6/15 10:10
import codecs

import torch

from config.config import DEVICE, DEFAULT_CONFIG
import json

# f = open('../data/result.txt', 'r')
# examples = []
# for line in f.readlines():
#     examples.append(json.loads(line))
# f.close()
# with open(config.result_path, 'a', encoding='utf-8') as f:
#     for example in examples:
#         text = example['originalText']
#         entities = example['entities']
#         f.write('{' + '"originalText": "{}", "entities": ['.format(text))
#         for i, entity in zip(range(len(entities)), entities):
#             if i == len(entities) - 1:
#                 f.write('{' + '"start_pos": {}, "end_pos": {}, "label_type": "{}"'.format(entity['start_pos'],
#                                                                                           entity['end_pos'],
#                                                                                           entity['label_type'])
#                         + '}]}')
#             else:
#                 f.write('{' + '"start_pos": {}, "end_pos": {}, "label_type": "{}"'.format(entity['start_pos'],
#                                                                                           entity['end_pos'],
#                                                                                           entity['label_type'])
#                         + '}, ')
#         f.write('\n')
import numpy
from tqdm import tqdm
from utils.tool import tool
from module.module import CCKS2020_NER
from config.config import DEVICE

# train_data_json2list = tool.read_json(config.task1_all_path)
# train_data = tool.load_data(train_data_json2list)
#
# word_vocab = tool.get_text_vocab(train_data)
# tag_vocab = tool.get_tag_vocab(train_data)
#
# test_data = tool.read_json(config.task1_all_path)
# i = 0
#
# with open(config.result_path, 'a', encoding='utf-8') as f:
#     for example in tqdm(test_data):
#         sent = example['originalText']
#         text = torch.tensor(
#             numpy.array([word_vocab.stoi[word] for word in sent], dtype='int64')).unsqueeze(1).to(DEVICE)
#         text_len = torch.tensor(numpy.array([text.size(0)], dtype='int64')).to(DEVICE)
#         pass
# is_gpu = torch.cuda.is_available()
# gpu_num = torch.cuda.device_count()
# gpu_name = torch.cuda.get_device_name(0)
# gpu_index = torch.cuda.current_device()
# import torch
# from torch import nn
#
#
# class DiceLoss(nn.Module):
#     """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
#     Useful in dealing with unbalanced data
#     Add softmax automatically
#     """
#
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, y_pred, y_true):
#         # shape(y_pred) = batch_size, label_num, **
#         # shape(y_true) = batch_size, **
#         y_pred = torch.softmax(y_pred, dim=1)
#         pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))
#         dsc_i = 1 - ((1 - pred_prob) * pred_prob) / ((1 - pred_prob) * pred_prob + 1)
#         dice_loss = dsc_i.mean()
#         return dice_loss
#
#
# # =================================TEST=================================
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc = nn.Linear(4, 32)
#         self.fc2 = nn.Linear(32, 3)
#
#     def forward(self, x):
#         return self.fc2(self.fc(x))
#
#
# if __name__ == '__main__':
#
#     def test():
#         from sklearn.datasets import load_iris
#         from sklearn.model_selection import train_test_split
#         data = load_iris()
#         X = data['data']
#         y = data['target']
#         model = Net()
#         opti = torch.optim.Adam(model.parameters())
#         criterion = DiceLoss()
#         from torch.utils.data import TensorDataset, DataLoader
#         tx, cvx, ty, cvy = train_test_split(X, y, test_size=0.2)
#         train_loader = DataLoader(TensorDataset(torch.FloatTensor(tx), torch.LongTensor(ty)), batch_size=32)
#         dev_loader = DataLoader(TensorDataset(torch.FloatTensor(cvx), torch.LongTensor(cvy)), batch_size=32)
#         for i in range(100):
#             for train_x, train_y in train_loader:
#                 output = model(train_x)
#                 loss = criterion(output, train_y)
#                 opti.zero_grad()
#                 loss.backward()
#                 opti.step()
#             for dev_x, dev_y in dev_loader:
#                 output = model(dev_x)
#                 true_num = (output.argmax(dim=1) == dev_y).sum()
#             print(true_num)
#
#
#     test()
# from utils.build_word2vec_weights import load_vec
# vec = load_vec('D:/ZUTNLP/zutnlp/medical/baidubaike/baidubaike.bigram-char', 300)
# info = {'epoch': [], 'p': [], 'r': [], 'f1': [], 'loss': []}
# for i in range(100):
#     num = (100 - i) * (100 - i)
#     f1 = i
#     p = i + 1
#     info['epoch'].append(i)
#     info['loss'].append(num)
#     info['f1'].append(f1)
#     info['p'].append(p)
#
import matplotlib.pyplot as plt

#
# plt.plot(info['epoch'], info['loss'], label='loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('loss.png')
# plt.cla()
# plt.plot(info['epoch'], info['p'], label='p')
# plt.plot(info['epoch'], info['f1'], label='f1')
# plt.legend()
# plt.savefig('p_r_f1_score.png')

# plt.show()

# lamda = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
# f1 = [0.823, 0.827, 0.825, 0.828, 0.831, 0.832, 0.827, 0.825, 0.827, 0.823]
# plt.plot(lamda, f1, label='F1-score')
# plt.xlabel('lambda')
# plt.ylabel('F1-score')
# plt.title('F1-score Graph')
# plt.show()
# x = numpy.array([[1, 2, 3], [4, 5, 6]])
# print(x[:, : 2])
from sklearn.model_selection import train_test_split

train_data_json2list, dev_data_json2list = train_test_split(tool.read_json('../data/split_data/train_all.txt'),
                                                            test_size=0.8, random_state=42)
# train_data = tool.load_data(tool.read_json(DEFAULT_CONFIG['train_path']))
# dev_data = tool.load_data(tool.read_json(DEFAULT_CONFIG['dev_path']))
train_data = tool.load_data(train_data_json2list)
dev_data = tool.load_data(dev_data_json2list)
examples = train_data.examples
with codecs.open('../data/bert/train.txt', 'w', encoding='utf-8') as f:
    for example in examples:
        text_list = example.text
        tag_list = example.tag
        assert len(text_list) == len(tag_list)
        for ch, tag in zip(text_list, tag_list):
            f.write(ch + '\t' + tag + '\n')
            # print(ch, tag)
        f.write('\n')
'''
transfer
len:    >= 150  2  delete
        >= 200  1  delete
'''
# with codecs.open('../data/bert/transfer_train.txt', 'w', encoding='utf-8') as f:
#     for example in examples:
#         text_list = example.text
#         tag_list = example.tag
#         assert len(text_list) == len(tag_list)
#         if len(text_list) >= 150:
#             continue
#         for ch, tag in zip(text_list, tag_list):
#             f.write(ch + '\t' + tag[0] + '\n')
#         f.write('\n')
examples = dev_data.examples
with codecs.open('../data/bert/dev.txt', 'w', encoding='utf-8') as f:
    for example in examples:
        text_list = example.text
        tag_list = example.tag
        assert len(text_list) == len(tag_list)
        for ch, tag in zip(text_list, tag_list):
            f.write(ch + '\t' + tag + '\n')
            # print(ch, tag)
        f.write('\n')
'''
transfer
'''
# with codecs.open('../data/bert/transfer_dev.txt', 'w', encoding='utf-8') as f:
#     for example in examples:
#         text_list = example.text
#         tag_list = example.tag
#         assert len(text_list) == len(tag_list)
#         if len(text_list) >= 150:
#             continue
#         for ch, tag in zip(text_list, tag_list):
#             f.write(ch + '\t' + tag[0] + '\n')
#         f.write('\n')
# f = open('../data/vocab/entity_vocab.txt', 'r', encoding='utf-8')
# lines = f.readlines()
# f.close()
# for index, line in enumerate(lines):
#     if len(line.strip()) == 1:
#         print(index, line.strip())
pass
