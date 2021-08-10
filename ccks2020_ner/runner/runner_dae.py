#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_dae
# @Author   : 研哥哥
# @Time     : 2020/7/17 15:39

import sys

sys.path.append('../')
from config.config import DEFAULT_CONFIG
from module.module import CCKS2020_NER

if __name__ == '__main__':
    DEFAULT_CONFIG['model_name'] = 'transformer_bilstm_crf'
    DEFAULT_CONFIG['experiment_name'] = '10_07_1744-b-t_d-dae_1.2-dice_0.01-result_valid'
    DEFAULT_CONFIG['save_model_path'] = DEFAULT_CONFIG['save_model_path'].format(DEFAULT_CONFIG['model_name'])
    DEFAULT_CONFIG['result_path'] = DEFAULT_CONFIG['result_path'].format(DEFAULT_CONFIG['model_name'])
    ccks2020_ner = CCKS2020_NER()
    ccks2020_ner.train()
    ccks2020_ner.predict()
