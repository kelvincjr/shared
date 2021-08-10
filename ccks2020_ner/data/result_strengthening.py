#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : result_strengthening
# @Author   : 研哥哥
# @Time     : 2020/6/19 16:15

import codecs
import json

from module.module import CCKS2020_NER
from utils.tool import tool
from utils.build_elt_vocab import build_elt_vocab


def result_entity_strengthening():
    f = open('../result/test2/result_2-89.29.txt', 'r', encoding='utf-8')
    examples = []
    for line in f.readlines():
        examples.append(json.loads(line))
    f.close()
    label2entity = build_elt_vocab()
    i = ii = iii = 0
    result_list = []
    for example in examples:
        originalText = example['originalText']
        entities = example['entities']
        entity_list = []
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            label_type = entity['label_type']
            original_entity = originalText[start_pos: end_pos]
            if original_entity not in label2entity[label_type]:
                # print(original_entity, label_type)
                i += 1
                for label in label2entity:
                    if original_entity in label2entity[label]:
                        entity['label_type'] = label
                        print(original_entity, label)
                        ii += 1
            else:
                iii += 1
            entity_list.append(entity)
            # if original_entity in label2entity[label_type]:
            #     entity_list.append(entity)
        result_list.append({'originalText': originalText, 'entities': entity_list})
    with codecs.open('../result/test2/result_2_.txt', 'w', encoding='utf-8') as f:
        for result in result_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def result_merge():
    base_path = '../result/test2/result_{}.txt'
    examples_1 = tool.read_json(base_path.format(89.7952))
    examples_2 = tool.read_json(base_path.format(89.7366))
    examples_3 = tool.read_json(base_path.format(89.8467))
    examples_4 = tool.read_json(base_path.format(89.2934))
    examples_5 = tool.read_json(base_path.format(89.2062))
    examples_6 = tool.read_json(base_path.format(89.7952))
    examples = []
    for example_1, example_2, example_3, example_4, example_5, example_6 in zip(
            examples_1, examples_2, examples_3, examples_4, examples_5, examples_6
    ):
        originalText = example_1['originalText']
        entities_1 = example_1['entities']
        entities_2 = example_2['entities']
        entities_3 = example_3['entities']
        entities_4 = example_4['entities']
        entities_5 = example_5['entities']
        entities_6 = example_6['entities']
        entity_list = []
        for entity_1 in entities_1:
            label_type = entity_1['label_type']
            if label_type == '疾病和诊断':
                entity_list.append(entity_1)
        for entity_2 in entities_2:
            label_type = entity_2['label_type']
            if label_type == '影像检查':
                entity_list.append(entity_2)
        for entity_3 in entities_3:
            label_type = entity_3['label_type']
            if label_type == '实验室检验':
                entity_list.append(entity_3)
        for entity_4 in entities_4:
            label_type = entity_4['label_type']
            if label_type == '手术':
                entity_list.append(entity_4)
        for entity_5 in entities_5:
            label_type = entity_5['label_type']
            if label_type == '药物':
                entity_list.append(entity_5)
        for entity_6 in entities_6:
            label_type = entity_6['label_type']
            if label_type == '解剖部位':
                entity_list.append(entity_6)
        examples.append({'originalText': originalText, 'entities': entity_list})
    return examples
    pass


if __name__ == '__main__':
    ccks2020_ner = CCKS2020_NER()
    examples = result_merge()
    data = tool.load_data(examples)
    examples = data.examples
    with codecs.open('../result/uncommitted/result_test2.txt', 'w', encoding='utf-8') as f:
        for example in examples:
            text = ''.join(example.text)
            tag = example.tag
            entities = ccks2020_ner._build_list_dict(len(tag), tag)
            pred_dict = {'originalText': text, 'entities': entities}
            f.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
    pass
# print(sum_)
# vocab_list = []
# with open('./task1_vocab.val.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         vocab = line.split('\t')
#         vocab_dict = {'entity': vocab[0].strip(), 'label_type': vocab[1].strip()}
#         vocab_list.append(vocab_dict)
# label_stat = {'疾病和诊断': {'sum': 0, 'mix_len': 100, 'max_len': 0},
#               '影像检查': {'sum': 0, 'mix_len': 100, 'max_len': 0},
#               '实验室检验': {'sum': 0, 'mix_len': 100, 'max_len': 0},
#               '手术': {'sum': 0, 'mix_len': 100, 'max_len': 0},
#               '药物': {'sum': 0, 'mix_len': 100, 'max_len': 0},
#               '解剖部位': {'sum': 0, 'mix_len': 100, 'max_len': 0}}
# for vocab_dict in vocab_list:
#     label_stat[vocab_dict['label_type']]['sum'] += 1
#     entity_len = len(vocab_dict['entity'])
#     if entity_len > label_stat[vocab_dict['label_type']]['max_len']:
#         label_stat[vocab_dict['label_type']]['max_len'] = entity_len
#     if entity_len < label_stat[vocab_dict['label_type']]['mix_len']:
#         label_stat[vocab_dict['label_type']]['mix_len'] = entity_len
#
# example_index = 0
# entity_index = 0
# sum_ = 0
# result_list = []
# for example in examples:
#     example_index += 1
#     entity_index = 0
#     originalText = example['originalText']
#     entities = example['entities']
#     entity_list = []
#     for entity in entities:
#         entity_index += 1
#         bool_ = False
#         start_pos = entity['start_pos']
#         end_pos = entity['end_pos']
#         label_type = entity['label_type']
#         original_entity = originalText[start_pos: end_pos]
#         original_entity_len = end_pos - start_pos
#         for vocab_dict in vocab_list:
#             if label_type == vocab_dict['label_type'] and \
#                     original_entity == vocab_dict['entity'] and \
#                     original_entity_len == len(vocab_dict['entity']):
#                 bool_ = True
#                 # print(original_entity, vocab_dict['entity'], label_type)
#         if not bool_:
#             for vocab_dict in vocab_list:
#                 if originalText[start_pos: start_pos + len(vocab_dict['entity'])] == vocab_dict['entity'] and \
#                         label_type == vocab_dict['label_type']:
#                     entity['end_pos'] = start_pos + len(vocab_dict['entity'])
#                     sum_ += 1
#                     print(original_entity, vocab_dict['entity'], label_type, example_index, entity_index)
#         entity_list.append(entity)
#     result_list.append({'originalText': originalText, 'entities': entity_list})