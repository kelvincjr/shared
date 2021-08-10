#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : config_msra
# @Author   : 研哥哥
# @Time     : 2020/7/15 21:25

DEFAULT_CONFIG = {
    'model_name': None,
    'experiment_name': None,
    'train_path': '../data/msra_3_bioes/bioes.msra.train.ner',
    'dev_path': '../data/msra_3_bioes/bioes.msra.test.ner',
    'test_path': None,
    'result_path': None,
    'save_path': '../model/save_model/',
    'pred_info_path': '../result/{}_pred_info.txt',
    'tag_type': 'BME_SO',  # BIO or BME_SO
    'use_cuda': True,
    'epoch': 100,
    'batch_size': 64,
    'learning_rate': 0.0002,
    'num_layers': 2,
    'pad_index': 1,
    'n_hid': 200,  # the dimension of the feedforward network model in nn.TransformerEncoder
    'n_layers': 2,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    'n_head': 2,  # the number of heads in the multiheadattention models
    'dropout': 0.5,  # the dropout value
    'attn_dropout': 0.2,
    'key_dim': 64,
    'val_dim': 64,
    'num_heads': 3,
    'use_attn': True,
    'embedding_dim': 768,  # embedding dimension     预训练模型：hidden 1024/786   word2voc：300
    'hidden_dim': 300,

    'use_dae': True,
    'lm_lamda': 1,
    'use_dice': False,
    'dice_lamda': 0.01,
    'use_vectors': True,
    # 'vector_win_path': 'D:/ZUTNLP/zutnlp/medical/baidubaike/baidubaike.bigram-char',
    'vector_win_path': 'D:/ZUTNLP/zutnlp/medical/bertvec/bert_vectors_768.txt',
    # 'vector_linux_path': '/home/zutnlp/liuyan/CCKS_EE/vector/baidubaike.bigram_char',
    'vector_linux_path': '/home/zutnlp/data/bertvec/bert_vectors_768.txt',
}
