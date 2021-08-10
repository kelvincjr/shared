#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_bert_bilstm_crf
# @Author   : 研哥哥
# @Time     : 2020/7/29 17:41

from config.config import DEFAULT_CONFIG
from module.module import CCKS2020_NER

if __name__ == '__main__':
    DEFAULT_CONFIG['model_name'] = 'transformer_bilstm_crf'
    DEFAULT_CONFIG['experiment_name'] = 'transformer_bilstm_crf-split_3-bert_768-lam_5.0'
    DEFAULT_CONFIG['pred_info_path'] = DEFAULT_CONFIG['pred_info_path'].format(DEFAULT_CONFIG['model_name'],
                                                                               DEFAULT_CONFIG['experiment_name'])
    ccks2020_ner = CCKS2020_NER()
    ccks2020_ner.train()
    # ccks2020_ner.predict()
