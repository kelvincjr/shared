#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : build_elt_vocab
# @Author   : 研哥哥
# @Time     : 2020/7/22 12:02
import codecs

from utils.tool import tool
from config.config import DEFAULT_CONFIG


def build_elt_vocab():
    # examples = tool.read_json(DEFAULT_CONFIG['train_path'])
    examples = tool.read_json('../data/ccks2020_ner/task1_train.txt')
    label_stat = {'疾病和诊断': [],
                  '影像检查': [],
                  '实验室检验': [],
                  '手术': [],
                  '药物': [],
                  '解剖部位': []}

    for example in examples:
        originalText = example['originalText']
        entities = example['entities']
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            label_type = entity['label_type']
            e = originalText[start_pos: end_pos]
            if e not in label_stat[label_type]:
                label_stat[label_type].append(e)
    for vocab_path in DEFAULT_CONFIG['vocab_path']:
        f = open(vocab_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            e, label = line.strip().split('\t')
            if e not in label_stat[label]:
                label_stat[label].append(e)
    return label_stat


def build_entity_vocab():
    examples = tool.read_json('../data/ccks2020_ner/task1_train.txt')
    vocab = []
    entity_num_dict = {}
    for example in examples:
        originalText = example['originalText']
        entities = example['entities']
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            e = originalText[start_pos: end_pos]
            if e not in vocab:
                vocab.append(e)
                if e == '异丙嗪':
                    print(e)
                # if len(e) == 1:
                #     print(e)
            if e not in entity_num_dict:
                entity_num_dict[e] = 1
            else:
                entity_num_dict[e] += 1
    vocab_paths = ['../data/vocab/task1_vocab.txt', '../data/vocab/task1_vocab.val.txt']
    # for vocab_path in DEFAULT_CONFIG['vocab_path']:
    for vocab_path in vocab_paths:
        f = open(vocab_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            e, label = line.strip().split('\t')
            if e not in vocab:
                vocab.append(e)
            if e not in entity_num_dict:
                entity_num_dict[e] = 1
            else:
                entity_num_dict[e] += 1
    for index, entity in enumerate(entity_num_dict):
        if entity_num_dict[entity] == 1:
            print(index, entity)

    # print(entity_num_dict['：胃'])
    return vocab, entity_num_dict


def vocab2txt(path='../data/vocab/entity_vocab.txt', vocab=None):
    with codecs.open(path, 'w', encoding='utf-8') as f:
        for v in vocab:
            f.write(v + '\n')


def vocab2txt_num(path='../data/vocab/entity_num_vocab.txt', vocab=None):
    with codecs.open(path, 'w', encoding='utf-8') as f:
        for v in vocab:
            f.write(v + '\t' + str(vocab[v]) + '\n')


if __name__ == '__main__':
    # label2entity = build_elt_vocab()
    # for label in label2entity:
    #     entity_list = label2entity[label]
    #     for e in entity_list:
    #         for l in label2entity:
    #             if e in label2entity[l] and label != l:
    #                 print(e, label, l)
    # pass
    entity_vocab, entity_num_dict = build_entity_vocab()
    vocab2txt(vocab=entity_vocab)
    vocab2txt_num(vocab=entity_num_dict)
    pass
