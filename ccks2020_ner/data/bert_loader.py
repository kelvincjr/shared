#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : bert_loader
# @Author   : 研哥哥
# @Time     : 2020/7/29 17:32

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data.bert_processor import MyPro, convert_examples_to_features
from utils.log import logger


def init_params(vocab_path):
    processor = MyPro()
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    return processor, tokenizer


def create_batch_iter(mode, config, processor, tokenizer, label_vocab=None, mask_vocab=None):
    """构造迭代器"""
    train_path = config['train_path']
    dev_path = config['dev_path']
    train_batch_size = config['batch_size']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    num_train_epochs = config['epoch']
    eval_batch_size = config['batch_size']
    max_seq_length = config['max_seq_len']
    split_tag = config['split_tag']

    # processor, tokenizer = init_params(vocab_path)

    if mode == "train":
        examples = processor.get_train_examples(path=train_path, split=split_tag)
        num_train_steps = int(len(examples) / train_batch_size / gradient_accumulation_steps * num_train_epochs)
        batch_size = train_batch_size
        logger.info("  Num steps = %d", num_train_steps)

    elif mode == "dev":
        examples = processor.get_dev_examples(path=dev_path, split=split_tag)
        batch_size = eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    # label_vocab = processor.get_label_vocab()
    # mask_vocab = processor.get_mask_vocab(path=vocab_path)

    # 特征
    features = convert_examples_to_features(examples, label_vocab, mask_vocab, max_seq_length, tokenizer)

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "dev":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == "dev":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)


if __name__ == '__main__':
    num_train_steps = int(38 / 32 / 1 * 5)
    print(num_train_steps)
