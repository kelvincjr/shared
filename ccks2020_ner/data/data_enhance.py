#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_enhance
# @Author   : 研哥哥
# @Time     : 2020/8/8 1:03
import codecs
import json

from utils.tool import tool


def find_all(example=None, examples=None):
    for e in examples:
        if e['originalText'] == example['originalText']:
            return False

    return True


def find_val(example=None, examples=None):
    examples_s = []
    for e in examples:
        if e['originalText'] == example['originalText']:
            e['entities'] = example['entities']
        examples_s.append(e)
    return examples_s


def data_enhance(path=None):
    sum_1 = sum_2 = 0
    examples = tool.read_json('./ccks2020_ner/task1_train_.txt')
    examples_1 = tool.read_json('./ccks2020_ner/task1_train_1.txt')
    examples_2 = tool.read_json('./ccks2020_ner/task1_train_2.txt')
    examples_3 = tool.read_json('./ccks2020_ner/task1_train_3.txt')
    examples_4 = tool.read_json('./ccks2020_ner/task1_no_val.txt')

    for example_4 in examples_4:
        if find_all(example_4, examples):
            sum_1 += 1
        else:
            sum_2 += 1
            # print(example_2['originalText'])

    for example_2 in examples_2:
        if find_all(example_2, examples_1):
            sum_1 += 1
            examples_1.append(example_2)
        else:
            sum_2 += 1
            # print(example_2['originalText'])
    print(sum_1, sum_2)
    for example_3 in examples_3:
        if find_all(example_3, examples_1):
            sum_1 += 1
            examples_1.append(example_3)
        else:
            sum_2 += 1
            # print(example_3['originalText'])
    print(sum_1, sum_2)

    for example_2 in examples_2:
        examples_1 = find_val(example_2, examples_1)
    for example_3 in examples_3:
        examples_1 = find_val(example_3, examples_1)

    with codecs.open('./ccks2020_ner/task1_train_.txt', 'w', encoding='utf-8') as f:
        for result in examples_1:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    data_enhance()
