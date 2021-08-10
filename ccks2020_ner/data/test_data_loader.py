#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : test_data_loader
# @Author   : 研哥哥
# @Time     : 2020/6/12 17:50

import codecs
import json

from sklearn.model_selection import train_test_split
from config.config import DEFAULT_CONFIG
from utils.tool import tool

# len_stat = {'0': 0,
#             '50': 0,
#             '100': 0,
#             '150': 0,
#             '200': 0,
#             '300': 0}
# path = DEFAULT_CONFIG['unlabeled_path']
# if type(path) is list:
#     print(1)
# examples = []
# total_len_1 = total_len_2 = 0
# for p in path:
#     f = open(p, 'r', encoding='utf-8')
#     lines = f.readlines()
#     f.close()
#     for line in lines:
#         line = line.strip()
#         total_len_1 += len(line)
#         start = split_len = 0
#         for index, ch in enumerate(line):
#             split_len += 1
#             # if ch == '。' or ch == '；':
#             if ch == '。':
#                 examples.append(line[start: index + 1])
#                 start = index + 1
#                 split_len = 0
#             elif split_len >= 100:
#                 if ch == ',' or ch == '，' or ch == '；':
#                     examples.append(line[start: index + 1])
#                     start = index + 1
#                     split_len = 0
#             elif index + 1 == len(line):
#                 split_line = line[start: index + 1]
#                 if len(split_line) < 10:
#                     print(1)
#                 examples.append(split_line)
#
# for example in examples:
#     total_len_2 += len(example)
#     text_len = len(example)
#     if 0 <= text_len < 50:
#         len_stat['0'] += 1
#     elif 50 <= text_len < 100:
#         len_stat['50'] += 1
#     elif 100 <= text_len < 150:
#         len_stat['100'] += 1
#     elif 150 <= text_len < 200:
#         len_stat['150'] += 1
#     elif 200 <= text_len < 300:
#         len_stat['200'] += 1
#     else:
#         len_stat['300'] += 1
'''
dev: task1_train task1_all test2
train_dev: train_test 0.05
'''
train_data_json2list, dev_data_json2list = train_test_split(
    tool.read_json('./ccks2020_ner/result_90.0921.txt'), test_size=0.05
)
with codecs.open('./ccks2020_ner/result_train.txt', 'w', encoding='utf-8') as f:
    for result in train_data_json2list:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
with codecs.open('./ccks2020_ner/result_valid.txt', 'w', encoding='utf-8') as f:
    for result in dev_data_json2list:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
pass
