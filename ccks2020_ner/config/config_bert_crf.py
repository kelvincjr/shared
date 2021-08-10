#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : config_bert_crf
# @Author   : 研哥哥
# @Time     : 2020/6/24 9:12

import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

DEFAULT_CONFIG = {
    'model_name': None,
    'experiment_name': None,
    'train_path': '../data/bert/split_train.txt',
    'dev_path': '../data/bert/split_dev.txt',
    'test_path': '../data/task1_no_val.txt',
    'unlabeled_path': ['../data/task1_unlabeled.txt', '../data/task1_unlabeled.val.txt'],
    'vocab_path': '../data/task1_vocab.txt',
    'result_path': '../result/result.txt',
    'save_path': '../model/save_model/',
    'pred_info_path': '../result/{}/{}_pred_info.txt',
    'tag_type': 'BME_SO',  # BIO or BME_SO
    'use_cuda': True,
    'epoch': 50,
    'batch_size': 10,
    'learning_rate': 2e-5,
    'lr_decay': 0.001,
    'tag_num': 0,
    'max_seq_len': 250,
    'split_tag': '\t',
    'optimizer': 'Adam',  # 默认提供两个（Adam或SGD）
    'num_layers': 1,
    'pad_index': 1,
    'dropout': 0.5,  # the dropout value
    'attn_dropout': 0.2,
    'embedding_dim': 768,  # embedding dimension     词嵌入: BERT_768 Random_300
    'hidden_dim': 300,

    'use_lstm': True,
    'use_dae': False,
    'lm_lamda': 1.2,
    'use_dice': False,
    'dice_lamda': 0.01,
    'use_vectors': True,
    # 'vector_win_path': 'D:/ZUTNLP/zutnlp/medical/baidubaike/baidubaike.bigram-char',
    'vector_win_path': 'D:/ZUTNLP/zutnlp/medical/bertvec/bert_vectors_768.txt',
    # 'vector_linux_path': '/home/zutnlp/liuyan/CCKS_EE/vector/baidubaike.bigram_char',
    'vector_linux_path': '/home/zutnlp/data/bertvec/bert_vectors_768.txt',
    # 'bert_path': 'D:/ZUTNLP/data/bert-base-chinese',
    'bert_path': '/home/zutnlp/data/bert-base-chinese',
    # 'vocab_path': 'D:/ZUTNLP/data/bert-base-chinese/vocab.txt',  # 词典路径
    'vocab_path': '/home/zutnlp/data/bert-base-chinese/vocab.txt',  # 词典路径
    'gradient_accumulation_steps': 1,  # 梯度堆积台阶
    'warmup_proportion': 0.1,
}
