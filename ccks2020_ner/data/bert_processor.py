#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : bert_processor
# @Author   : 研哥哥
# @Time     : 2020/7/29 17:32

import torch
from utils.corpus_util import read_line_word_tag
from utils.log import logger


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask = output_mask


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""

    def get_train_examples(self, path, split="\t"):
        """读取训练集 """
        raise NotImplementedError()

    def get_dev_examples(self, path, split="\t"):
        """读取验证集"""
        raise NotImplementedError()

    def get_label_vocab(self, path, split="\t"):
        """获取标签字典"""
        raise NotImplementedError()

    def get_mask_vocab(self, path):
        """获取mask词典"""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as fr:
            lines = []
            for line in fr:
                _line = line.strip('\n')
                lines.append(_line)
            return lines


class MyPro(DataProcessor):
    """将数据构造成example格式"""

    def __init__(self):
        super(MyPro, self).__init__()
        self.label_vocab = None

    def _get_examples(self, path, set_type="train", split="\t"):
        examples = []
        text_a_list, labels_list, word_vocab, label_vocab = read_line_word_tag(path, split)
        num = 0
        for text_a, label in zip(text_a_list, labels_list):
            num += 1
            guid = "%s-%d" % (set_type, num)
            example = InputExample(guid=guid, text_a=text_a, label=label)
            examples.append(example)
        logger.info('success loading dataset from {}....'.format(path))
        return examples, label_vocab

    def get_train_examples(self, path, split="\t"):
        examples, label_vocab = self._get_examples(path, split=split)
        self.label_vocab = label_vocab
        return examples

    def get_dev_examples(self, path, split="\t"):
        examples, _ = self._get_examples(path, set_type="dev", split=split)
        return examples

    def get_label_vocab(self, path, split="\t"):
        _, _, word_vocab, label_vocab = read_line_word_tag(path, split)
        return word_vocab, label_vocab

    def get_mask_vocab(self, path):
        mask_vocab = {}
        with open(path, 'r', encoding="utf-8") as fr:
            for line in fr:
                _line = line.strip('\n')
                if "##" in _line and mask_vocab.get(_line) is None:
                    mask_vocab[_line] = 1
        return mask_vocab


def convert_examples_to_features(examples, label_vocab, mask_vocab, max_seq_length, tokenizer):
    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        # tokens_a = example.text_a
        # labels = example.label.split()
        labels = example.label

        if len(tokens_a) == 0 or len(labels) == 0:
            continue

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            labels = labels[:(max_seq_length - 2)]
        # ----------------处理source--------------
        # 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        # 词转换成数字
        # input_ids = [mask_vocab[s] for s in tokens]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # ---------------处理target----------------
        # Notes: label_id中不包括[CLS]和[SEP]
        label_id = [label_vocab[l] for l in labels]
        label_padding = [-1] * (max_seq_length - len(label_id))
        label_id += label_padding

        # output_mask用来过滤bert输出中sub_word的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
        # 此外，也是为了适应crf
        output_mask = [0 if mask_vocab.get(t) is not None else 1 for t in tokens_a]
        output_mask = [0] + output_mask + [0]
        output_mask += padding

        # if ex_index < 1:
        #     logger.info("-----------------Example-----------------")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("label: %s " % " ".join([str(x) for x in label_id]))
        #     logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
        # ----------------------------------------------------

        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_id=label_id,
                               output_mask=output_mask)
        features.append(feature)

    return features


def convert_example(examples, mask_vocab, max_seq_length, tokenizer):
    """预测时处理"""
    all_input_ids, all_input_mask, all_segment_ids, all_output_mask = [], [], [], []
    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        # ----------------处理source--------------
        # 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        # 词转换成数字
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        output_mask = [0 if mask_vocab.get(t) is not None else 1 for t in tokens_a]
        output_mask = [0] + output_mask + [0]  #
        output_mask += padding

        # print('input_ids:',input_ids)
        # print('input_mask:',input_mask)
        # print('segment_ids:',segment_ids)
        # print('out_mask:',output_mask)

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_output_mask.append(output_mask)

    all_input_ids = torch.LongTensor(all_input_ids)
    all_input_mask = torch.LongTensor(all_input_mask)
    all_segment_ids = torch.LongTensor(all_segment_ids)
    all_output_mask = torch.LongTensor(all_output_mask)

    return all_input_ids, all_input_mask, all_segment_ids, all_output_mask
