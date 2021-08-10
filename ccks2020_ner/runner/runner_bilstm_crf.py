#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_bilstm_crf
# @Author   : 研哥哥
# @Time     : 2020/6/21 20:59

import sys

sys.path.append('../')
from config.config import DEFAULT_CONFIG
from module.module import CCKS2020_NER

if __name__ == '__main__':
    DEFAULT_CONFIG['model_name'] = 'bilstm_crf'
    DEFAULT_CONFIG['experiment_name'] = 'train_dev-dice_0.01-dae_1.2-bert_768'
    DEFAULT_CONFIG['save_model_path'] = DEFAULT_CONFIG['save_model_path'].format(DEFAULT_CONFIG['model_name'])
    DEFAULT_CONFIG['result_path'] = DEFAULT_CONFIG['result_path'].format(DEFAULT_CONFIG['model_name'])
    ccks2020_ner = CCKS2020_NER()
    ccks2020_ner.train()
    ccks2020_ner.predict()
