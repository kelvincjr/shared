#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_split
# @Author   : 研哥哥
# @Time     : 2020/6/13 10:24
"""
82划分数据集
1. 多空格
2.
3. 。错位。
4. 重要遗留问题，本地看  与  程序读入  是否会造成真正错位？  待验证

split_train_
1. 按照句号划分
2. 按照句号 and 分号划分
3. 按照句号 and 大于150强制划分
4. 按照句号分号 and 大于50强制按照逗号顿号划分
5. 按照句号 and 大于50强制按照分号逗号顿号划分
6. 按照句号 and 大于100强制按照分号逗号顿号划分
7. 按照句号 and 大于50强制按照分号逗号顿号划分 : all -> split_train.txt
"""
import codecs
import json
import random
from utils.tool import tool
from config.config import *
from module.module import CCKS2020_NER


# f = open('./task1_all.txt', 'r', encoding='utf8')
# examples = []
# for line in f.readlines():
#     examples.append(json.loads(line))
# f.close()
#
# random.shuffle(examples)
#
# with open('./task1_train.txt', 'w', encoding='utf8') as train_txt:
#     for i in range(0, 850):
#         train_txt.write(str(examples[i]) + '\r')
# with open('./task1_dev.txt', 'w', encoding='utf8') as valid_txt:
#     for i in range(850, len(examples)):
#         valid_txt.write(str(examples[i]) + '\r')'./split_train.txt'

def _build_list_dict(_len, _list):
    build_list = []
    tag_dict = {'disease_and_diagnosis': '疾病和诊断',
                'check': '影像检查',
                'checkout': '实验室检验',
                'operation': '手术',
                'medicine': '药物',
                'anatomical_site': '解剖部位'}
    for index, tag in zip(range(_len), _list):
        if tag[0] == 'B':
            start_pos = index
            end_pos = index + 1
            label_type = tag[2:]
            while _list[end_pos][0] == 'I' and _list[end_pos][2:] == label_type:
                end_pos += 1
            build_list.append({'start_pos': start_pos,
                               'end_pos': end_pos,
                               'label_type': tag_dict[label_type]})
    return build_list


train_data_json2list = tool.read_json('./ccks2020_ner/result_valid.txt')
train_data = tool.load_data(train_data_json2list)
examples = train_data.examples
examples_s = []
split_str_1 = '。'
split_str_2 = '，,、；;'
for example in examples:
    text_list = example.text
    tag_list = example.tag
    start = end = split_len = 0
    for index, ch in enumerate(text_list):
        split_len += 1
        end += 1
        # if ch == '。' or ch == '；':
        if ch in split_str_1 and tag_list[index][0] == 'O':
            examples_s.append({'text': text_list[start: end], 'tag': tag_list[start: end]})
            start = end
            split_len = 0
        elif split_len >= 50 and ch in split_str_2 and tag_list[index][0] == 'O':
            # if ch == ',' or ch == '，' or ch == '；':
            examples_s.append({'text': text_list[start: end], 'tag': tag_list[start: end]})
            start = end
            split_len = 0
        elif end == len(text_list):
            if len(text_list[start: end]) < 10:
                print(1)
            examples_s.append({'text': text_list[start: end], 'tag': tag_list[start: end]})
sum1_text = sum1_tag = sum2_text = sum2_tag = max_len = 0
min_len = 10000
for example in examples:
    text_list = example.text
    tag_list = example.tag
    sum1_text += len(text_list)
    sum1_tag += len(tag_list)
    end_text = text_list[len(text_list) - 1]
    end_tag = tag_list[len(tag_list) - 1]
    if end_tag[0] == 'B':
        print('你最得乐呵的~~~', end_text, end_tag)
for example in examples_s:
    text_list = example['text']
    tag_list = example['tag']
    if len(text_list) > max_len:
        max_len = len(text_list)
    if len(text_list) < min_len:
        min_len = len(text_list)
    # if len(text_list) < 10:
    #     print(''.join(text_list))
    sum2_text += len(text_list)
    sum2_tag += len(tag_list)
    end_text = text_list[len(text_list) - 1]
    end_tag = tag_list[len(tag_list) - 1]
    if end_text != '。' and end_text != '；' and end_text != ',' and end_text != '，':
        print('得乐呵的~~~', end_text, end_tag)
    if end_tag[0] == 'B':
        print('你更得乐呵的~~~', end_text, end_tag)

with codecs.open('./split_data/result_valid.txt', 'w', encoding='utf-8') as f:
    for example in examples_s:
        text = ''.join(example['text'])
        tag_list = example['tag']
        entities = _build_list_dict(len(tag_list), tag_list)
        pred_dict = {'originalText': text, 'entities': entities}
        f.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
