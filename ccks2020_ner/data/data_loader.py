#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_loader
# @Author   : 研哥哥
# @Time     : 2020/6/12 17:43

import sys
import collections
from torchtext.data import Field, Dataset, Example, BucketIterator
from utils.log import logger
from config.config import DEFAULT_CONFIG


def x_tokenizer(sentence):
    return [word for word in sentence]
    # return sentence


def y_tokenizer(tag: str):
    return [tag]


TEXT = Field(sequential=True, use_vocab=True, tokenize=x_tokenizer, include_lengths=True)
TAG = Field(sequential=True, tokenize=y_tokenizer, use_vocab=True, is_target=True, pad_token=None)
BI_GRAM = Field(sequential=True, use_vocab=True, tokenize=x_tokenizer, include_lengths=True)
LATTICE = Field(sequential=True, use_vocab=True, tokenize=x_tokenizer, include_lengths=True)
fields = [('text', TEXT), ('tag', TAG)]
fields_fl = [('bi_gram', BI_GRAM), ('lattice', LATTICE), ('tag', TAG)]


class DataLoader(Dataset):
    def __init__(self, examples=None, **kwargs):
        self.encoding = 'utf-8'
        self.examples = examples
        if DEFAULT_CONFIG['model_name'] == 'flat_lattice':
            self.fields = fields_fl
        else:
            self.fields = fields
        self.examples = self.get_label(self.examples)
        super(DataLoader, self).__init__(self.examples, self.fields, **kwargs)

    def get_label(self, examples):
        label = []
        if DEFAULT_CONFIG['model_name'] == 'flat_lattice':
            f = open(DEFAULT_CONFIG['vocab_path'], 'r', encoding='utf-8')
            lines = f.readlines()
            f.close()
            w_list = []
            for line in lines:
                w = line.strip()
                w_list.append(w)
            w_trie = Trie()
            for w in w_list:
                w_trie.insert(w)
        for example in examples:
            originalText = example['originalText']
            tag_list = self.get_tag(example)
            if DEFAULT_CONFIG['model_name'] == 'flat_lattice':
                bi_gram, lattice = self.get_flat_lattice(originalText, w_trie)
                assert len(bi_gram) == len(tag_list)
                label.append(Example.fromlist((bi_gram, lattice, tag_list), fields=self.fields))
            else:
                text_list = [x for x in originalText]
                assert len(text_list) == len(tag_list)
                label.append(Example.fromlist((text_list, tag_list), fields=self.fields))
        return label

    def get_flat_lattice(self, originalText=None, w_trie=None):
        bi_gram = [originalText[i: i + 2] for i in range(len(originalText) - 1)]
        bi_gram.append(originalText[-1] + 'end')
        lattice = list(originalText) + w_trie.get_lexicon(originalText)
        return bi_gram, lattice

    def get_tag(self, example):
        originalText = example['originalText']
        entities = example['entities']
        tag_list = ['O' for i in range(len(originalText))]
        '''
        total : 1500
        疾病和诊断   : Disease and diagnosis   6211
        影像检查    : check       1490
        实验室检验   : checkout   1885
        手术  : operation         1327
        药物  : medicine/drug     2841
        解剖部位    : Anatomical site          12660
        BME S O（Begin, Medium, End, Single, Other）
        '''
        tag_dict = {'疾病和诊断': 'disease_and_diagnosis',
                    '影像检查': 'check',
                    '实验室检验': 'checkout',
                    '手术': 'operation',
                    '药物': 'medicine',
                    '解剖部位': 'anatomical_site'}
        if DEFAULT_CONFIG['tag_type'] == 'BIO':
            for entity in entities:
                start_pos = entity['start_pos']
                end_pos = entity['end_pos']
                label_type = entity['label_type']
                tag_end = tag_dict[label_type]
                tag_list[start_pos] = 'B_' + tag_end
                for i in range(start_pos + 1, end_pos):
                    tag_list[i] = 'I_' + tag_end
        elif DEFAULT_CONFIG['tag_type'] == 'BME_SO':
            for entity in entities:
                start_pos = entity['start_pos']
                end_pos = entity['end_pos']
                label_type = entity['label_type']
                tag_end = tag_dict[label_type]
                if end_pos - start_pos == 1:
                    tag_list[start_pos] = 'S_' + tag_end
                else:
                    tag_list[start_pos] = 'B_' + tag_end
                    for i in range(start_pos + 1, end_pos - 1):
                        tag_list[i] = 'M_' + tag_end
                    tag_list[end_pos - 1] = 'E_' + tag_end
        else:
            logger.error('tag_type != BIO and tag_type != BME_SO !')
            sys.exit()
        return tag_list


class UnLabelDataLoader(Dataset):
    def __init__(self, examples=None, **kwargs):
        self.encoding = 'utf-8'
        self.examples = examples
        self.fields = fields
        self.examples = self.get_label(self.examples)
        super(UnLabelDataLoader, self).__init__(self.examples, self.fields, **kwargs)

    def get_label(self, examples):
        label = []
        for example in examples:
            originalText = example['originalText']
            text_list = [x for x in originalText]
            tag_list = ['O' for i in range(len(originalText))]
            assert len(text_list) == len(tag_list)
            label.append(Example.fromlist((text_list, tag_list), fields=fields))
        return label


class UnLabeledDataLoader(Dataset):
    def __init__(self, examples=None, **kwargs):
        self.encoding = 'utf-8'
        self.examples = examples
        self.fields = fields
        self.examples = self.get_label(self.examples)
        super(UnLabeledDataLoader, self).__init__(self.examples, self.fields, **kwargs)

    def get_label(self, examples):
        label = []
        for example in examples:
            text_list = [x for x in example]
            tag_list = ['O' for i in range(len(example))]
            assert len(text_list) == len(tag_list)
            label.append(Example.fromlist((text_list, tag_list), fields=fields))
        return label


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):
        current = self.root
        for c in w:
            current = current.children[c]
        current.is_w = True

    def search(self, w):
        current = self.root
        for c in w:
            current = current.children.get(c)
            if current is None:
                return -1
        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append(sentence[i:j + 1])

        return result


if __name__ == '__main__':
    # data_loader = DataLoader(input_file=config.train_path)
    data_loader = DataLoader(input_file='./task1_train.txt')
    examples = data_loader.read_json()
    print(len(examples))
