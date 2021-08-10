#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_stat
# @Author   : 研哥哥
# @Time     : 2020/6/19 16:16

import matplotlib.pyplot as plt
from utils.tool import tool
from data.data_loader import DataLoader
from config.config import DEFAULT_CONFIG
import json


def data_stat(examples=None, path='../data/ccks2020_ner/task1_train.txt'):
    if examples is None:
        examples = tool.read_json(path)
    label_stat = {'疾病和诊断': 0,
                  '影像检查': 0,
                  '实验室检验': 0,
                  '手术': 0,
                  '药物': 0,
                  '解剖部位': 0}
    len_stat = {'0': 0,
                '50': 0,
                '100': 0,
                '150': 0,
                '200': 0,
                '300': 0}
    max = 0
    for example in examples:
        originalText = example['originalText']
        text_len = len(originalText)
        if text_len > max:
            max = text_len
        if 0 <= text_len < 50:
            len_stat['0'] += 1
        elif 50 <= text_len < 100:
            len_stat['50'] += 1
        elif 100 <= text_len < 150:
            len_stat['100'] += 1
        elif 150 <= text_len < 200:
            len_stat['150'] += 1
        elif 200 <= text_len < 300:
            len_stat['200'] += 1
        else:
            len_stat['300'] += 1

        entities = example['entities']
        for entity in entities:
            label_stat[entity['label_type']] += 1
    return label_stat, len_stat


def find_str(path=None, text=None):
    examples = tool.read_json(path)
    for example in examples:
        originalText = example['originalText']
        entities = example['entities']
        if text == originalText:
            entity_s = []
            for entity in entities:
                start_pos = entity['start_pos']
                end_pos = entity['end_pos']
                label_type = entity['label_type']
                entity_s.append({'entity_name': originalText[start_pos: end_pos], 'label_type': label_type})
            print(originalText)
            print(entity_s)


# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))


if __name__ == '__main__':
    # name_list = ['疾病和诊断', '影像检查', '实验室检验', '手术', '药物', '解剖部位']
    # num_list = [4345, 1002, 1297, 923, 1935, 8811]
    # autolabel(plt.bar(range(len(num_list)), num_list, color='rgby', tick_label=name_list))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    # plt.title('医疗实体分布统计')
    # plt.xlabel('医疗实体类别')
    # plt.ylabel('所含实体数')
    # plt.show()
    # string = '。；，;,'
    # if '。' in string:
    #     print(1)
    # 5530 1303 1665 1224 2496 11329
    # >150: 111 7670  >200: 9150
    label_stat, len_stat = data_stat(path='../data/split_data/train_all.txt')
    print(label_stat)
    print(len_stat)
    # text = '入院前17年患者出现胸痛，呈闷痛，有时向肩背部放射痛，口服“速效救心丸”可在5分钟左右好转，夜间发作较多，活动后也可出现，曾于当地**医院就诊，口服阿司匹林、异搏定、美托洛尔等药物，此后长期门诊就诊，病情尚平稳。2013年12月再次出现胸闷不适，于外院行冠脉造影未见严重狭窄，建议药物治疗。入院前7小时患者突然出现胸闷，自服丹参症状无缓解，症状逐渐加重，持续不缓解，伴出汗、头晕，无心悸，无双下肢水肿，无头痛，无抽搐，无意识丧失，无大小便失禁，无血尿，由家人送入我院，做心电图提示\\\"心肌缺血\\\"，对症治疗后患者症状有所缓解，为进一步诊治以\\\"心绞痛\\\"收入院。患病以来患者食欲可，睡眠可，大小便正常，体重无明显变化。'
    # find_str('task1_train.txt', text=text)
