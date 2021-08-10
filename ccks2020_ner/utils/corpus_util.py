#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : corpus_util
# @Author   : 研哥哥
# @Time     : 2020/7/29 17:31


def read_line_word_tag(path, split="\t"):
    """
    读取数据，格式是：一行：一个字 一个标签
    :param path:
    :param split:
    :return:
    """
    text_a, label = "", ""
    text_list, label_list = [], []
    text_lines, label_lines = [], []
    word_vocab, label_vocab = {}, {}
    with open(path, encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            pairs = line.strip('\n').split(split)
            if len(pairs) == 2:
                word, tag = pairs[0], pairs[1]
                # 字典
                if word not in word_vocab:
                    word_vocab[word] = len(word_vocab)
                # 标签词典
                if tag not in label_vocab:
                    label_vocab[tag] = len(label_vocab)
                # 处理文本
                text_list.append(word)
                label_list.append(tag)
                text_a += word + " "
                label += tag + " "

            else:  # 如果是空行
                if len(text_list) >= 0 and len(label_list) > 0:
                    text_lines.append(text_list)
                    label_lines.append(label_list)
                text_a, label = "", ""  # text_a: 图 片 来 源 : 人 民 视 觉 # label：O O O O O O O O O
                text_list, label_list = [], []
    return text_lines, label_lines, word_vocab, label_vocab


if __name__ == '__main__':
    # path = "D:/ZUTNLP/data/ctb/ctb8/ner/ner_test.txt"
    path = '/home/zutnlp/datasets/ctb/ctb8/ner/ner_train.txt'
    text_lines, label_lines, label_vocab, word_vocab = read_line_word_tag(path)
    print(len(label_vocab), len(word_vocab))
