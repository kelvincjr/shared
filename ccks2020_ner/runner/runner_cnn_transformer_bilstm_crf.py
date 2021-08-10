#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_cnn_transformer_bilstm_crf
# @Author   : 研哥哥
# @Time     : 2020/8/6 11:25

import sys

sys.path.append('../')
from config.config import DEFAULT_CONFIG
from module.module import CCKS2020_NER

if __name__ == '__main__':
    DEFAULT_CONFIG['model_name'] = 'cnn_transformer_bilstm_crf'
    DEFAULT_CONFIG['experiment_name'] = 'cnn_transformer_bilstm_crf-split_3-bert_768-dae_1.0-dice_0.01'
    DEFAULT_CONFIG['save_model_path'] = DEFAULT_CONFIG['save_model_path'].format(DEFAULT_CONFIG['model_name'])
    DEFAULT_CONFIG['result_path'] = DEFAULT_CONFIG['result_path'].format(DEFAULT_CONFIG['model_name'])
    ccks2020_ner = CCKS2020_NER()
    ccks2020_ner.train()
    ccks2020_ner.predict()
