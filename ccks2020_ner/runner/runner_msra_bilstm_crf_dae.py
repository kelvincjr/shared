#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_msra_bilstm_crf_dae
# @Author   : 研哥哥
# @Time     : 2020/7/15 21:50

import sys

sys.path.append('../')
from config.config_msra import DEFAULT_CONFIG
from module.module_msra import CCKS2020_NER

if __name__ == '__main__':
    DEFAULT_CONFIG['model_name'] = 'bilstm_crf'
    DEFAULT_CONFIG['experiment_name'] = 'bilstm_crf-random-dae'
    DEFAULT_CONFIG['save_model_path'] = DEFAULT_CONFIG['save_model_path'].format(DEFAULT_CONFIG['model_name'])
    DEFAULT_CONFIG['result_path'] = DEFAULT_CONFIG['result_path'].format(DEFAULT_CONFIG['model_name'])
    ccks2020_ner = CCKS2020_NER()
    ccks2020_ner.train()
