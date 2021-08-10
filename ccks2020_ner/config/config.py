#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : config
# @Author   : 研哥哥
# @Time     : 2020/6/12 15:41

import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

DEFAULT_CONFIG = {
    # Base Config
    'model_name': None,
    'experiment_name': None,
    'train_path': '../data/split_data/task1_all-result_train.txt',  # ../data/split_data/train.txt or train_all.txt
    'dev_path': '../data/split_data/result_valid.txt',  # ../data/split_data/dev.txt
    'test_path': '../data/split_data/test2.txt',
    'unlabeled_path': ['../data/unlabeled_data/task1_unlabeled.txt', '../data/unlabeled_data/task1_unlabeled.val.txt'],
    #'vocab_path': ['../data/vocab/task1_vocab.txt', '../data/vocab/task1_vocab.val.txt'],
    'vocab_path': '../data/vocab/entity_vocab.txt',  # ['vocab/task1_vocab.txt', 'vocab/task1_vocab.val.txt']
    'save_model_path': '../model/save_model/{}/',
    'result_path': '../result/{}/',

    # Baseline Config
    'tag_type': 'BME_SO',  # BIO or BME_SO
    'use_cuda': True,
    'epoch': 100,
    'batch_size': 64,
    'learning_rate': 0.0002,
    'num_layers': 2,
    'pad_index': 1,
    'dropout': 0.5,  # the dropout value
    'embedding_dim': 768,  # embedding dimension     词嵌入: BERT_768 Random_300
    'hidden_dim': 768,
    'use_vectors': True,
    # 'vector_win_path': 'D:/ZUTNLP/zutnlp/medical/baidubaike/baidubaike.bigram-char',
    'vector_win_path': 'D:/ZUTNLP/data/word2vec/bertvec/bert_vectors_768.txt',
    'vector_linux_path': '../data/w2v/bert_vectors_768.txt',
    #'vector_linux_path': '/home/zutnlp/data/bertvec/bert_vectors_768.txt',

    # TransformerEncoder Config
    'n_hid': 200,  # the dimension of the feedforward network model in nn.TransformerEncoder
    'n_layers': 2,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    'n_head': 2,  # the number of heads in the multiheadattention models

    # CNN Config
    'chanel_num': 1,
    'filter_num': 100,

    # Attn Config
    'use_attn': False,
    'key_dim': 64,
    'val_dim': 64,
    'num_heads': 3,
    'attn_dropout': 0.5,

    # FlatLattice Config
    'bi_gram_embed_dim': 384,
    'lattice_embed_dim': 384,

    # RefactorLoss Config
    'use_dae': False,
    'dae_lambda': 1.2,
    'use_dice': True,
    'dice_lambda': 0.01,
}
