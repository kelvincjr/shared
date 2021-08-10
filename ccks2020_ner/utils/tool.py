#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : tool
# @Author   : 研哥哥
# @Time     : 2020/6/12 15:44
import codecs
import json
import matplotlib.pyplot as plt
from torchtext.datasets import SequenceTaggingDataset
from data.data_loader import *
from config.config import DEVICE


class Tool(object):
    def __int__(self):
        super(Tool, self).__init__()

    def load_data(self, examples=None):
        dataset = DataLoader(examples=examples)
        return dataset

    def load_unlabel_data(self, examples=None):
        dataset = UnLabelDataLoader(examples=examples)
        return dataset

    def load_unlabeled_data(self, examples=None):
        dataset = UnLabeledDataLoader(examples=examples)
        return dataset

    def load_msra_data(self, path=None):
        return SequenceTaggingDataset(path=path, fields=fields, separator=' ')

    def load_commodity_data(self, path=None):
        return SequenceTaggingDataset(path=path, fields=fields, separator=' ')

    def get_text_vocab(self, *dataset):
        TEXT.build_vocab(*dataset)
        return TEXT.vocab

    def get_bi_gram_vocab(self, *dataset):
        BI_GRAM.build_vocab(*dataset)
        return BI_GRAM.vocab

    def get_lattice_vocab(self, *dataset):
        LATTICE.build_vocab(*dataset)
        return LATTICE.vocab

    def get_tag_vocab(self, *dataset):
        TAG.build_vocab(*dataset)
        return TAG.vocab

    def get_iterator(self, dataset: Dataset, batch_size=64, sort_key=lambda x: len(x.text), sort_within_batch=True):
        iterator = BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                                  sort_within_batch=sort_within_batch, device=DEVICE)
        return iterator

    def get_lattice_iterator(self, dataset: Dataset, batch_size=64, sort_key=lambda x: len(x.lattice),
                             sort_within_batch=True):
        iterator = BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                                  sort_within_batch=sort_within_batch, device=DEVICE)
        return iterator

    def get_msra_iterator(self, dataset: Dataset, batch_size=64):
        return BucketIterator(dataset, batch_size=batch_size, shuffle=False, sort_key=lambda x: len(x.text),
                              sort_within_batch=True, device=DEVICE)

    def get_commodity_iterator(self, dataset: Dataset, batch_size=64):
        return BucketIterator(dataset, batch_size=batch_size, shuffle=False, sort_key=lambda x: len(x.text),
                              sort_within_batch=True, device=DEVICE)

    def read_json(self, path=None):
        f = open(path, 'r', encoding='utf-8')
        examples = []
        for line in f.readlines():
            examples.append(json.loads(line))
        f.close()
        return examples

    def save_json(self, examples: list, path: str):
        with codecs.open(path, 'w', encoding='utf-8') as f:
            for result in examples:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def read_unlabeled(self, paths=None):
        examples = []
        for path in paths:
            f = open(path, 'r', encoding='utf-8')
            lines = f.readlines()
            f.close()
            for line in lines:
                line = line.strip()
                start = split_len = 0
                for index, ch in enumerate(line):
                    split_len += 1
                    if ch == '。':
                        examples.append(line[start: index + 1])
                        start = index + 1
                        split_len = 0
                    elif split_len >= 100:
                        if ch == ',' or ch == '，' or ch == '；':
                            examples.append(line[start: index + 1])
                            start = index + 1
                            split_len = 0
                    elif index + 1 == len(line):
                        split_line = line[start: index + 1]
                        examples.append(split_line)
        return examples

    def adjust_learning_rate(self, optimizer=None, epoch=None):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = DEFAULT_CONFIG['learning_rate'] * (0.1 ** (epoch // 30))
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def record_pred_info(self, text=None, true_list=None, pred_list=None, path=None):
        pred_false = []
        un_pred = []
        for pred in pred_list:
            start_pos = pred['start_pos']
            end_pos = pred['end_pos']
            label_type = pred['label_type']
            _bool = False
            for true in true_list:
                if label_type == true['label_type'] and start_pos == true['start_pos'] and end_pos == true['end_pos']:
                    _bool = True
                    break
            if not _bool:
                pred_false.append({'entity': text[start_pos: end_pos], 'label_type': label_type})
        for true in true_list:
            start_pos = true['start_pos']
            end_pos = true['end_pos']
            label_type = true['label_type']
            _bool = False
            for pred in pred_list:
                if label_type == pred['label_type'] and start_pos == pred['start_pos'] and end_pos == pred['end_pos']:
                    _bool = True
                    break
            if not _bool:
                un_pred.append({'entity': text[start_pos: end_pos], 'label_type': label_type})
        pred_dict = {'originalText': text, 'pred_false': pred_false, 'un_pred': un_pred}
        with codecs.open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')

    def record_info2graph(self, info=None, result_path=None, experiment_name=None):
        # loss.png
        plt.plot(info['epoch'], info['loss'], label='loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss Graph')
        plt.savefig(result_path + '{}_loss.png'.format(experiment_name))
        plt.cla()
        # p_r_f1_score.png
        plt.plot(info['epoch'], info['p'], label='precision')
        plt.plot(info['epoch'], info['r'], label='recall')
        plt.plot(info['epoch'], info['f1'], label='f1')
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.title('Score Graph')
        plt.legend()
        plt.savefig(result_path + '{}_score.png'.format(experiment_name))


tool = Tool()
