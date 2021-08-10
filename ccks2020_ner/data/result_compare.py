#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : result_compare
# @Author   : LiuYan
# @Time     : 2020/9/27 14:36

from utils.tool import tool


def compare():
    result_1_path = '../result/test2/result_1-88.885.txt'
    result_2_path = '../result/test2/result_2-89.29.txt'
    result_1 = tool.read_json(path=result_1_path)
    result_2 = tool.read_json(path=result_2_path)
    i = 0
    result_list = list()
    for index, (result1, result2) in enumerate(zip(result_1, result_2)):
        entities_1 = build_entities(result1)
        entities_2 = build_entities(result2)
        lists = 0
        if len(entities_1) != len(entities_2):
            lists += 1
            i += 1
            if entities_2[0] in entities_1:
                pass
            pass
        result_list.append(lists)
    return result_2


def _compare(entities_1: list, entities_2: list):
    for entity in entities_1:
        if entity in entities_2:
            pass


def build_entities(example: dict):
    originalText = example['originalText']
    entities = example['entities']
    example_entities = list()
    for entity in entities:
        start_pos = entity['start_pos']
        end_pos = entity['end_pos']
        label_type = entity['label_type']
        e = originalText[start_pos: end_pos]
        example_entities.append(e)
    return example_entities


if __name__ == '__main__':
    result = compare()
    save_path = '../result/test2/result_2_.txt'
    tool.save_json(examples=result, path=save_path)
