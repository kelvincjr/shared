#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_bert_crf
# @Author   : 研哥哥
# @Time     : 2020/6/24 9:11

from config.config_bert_crf import DEFAULT_CONFIG
from module.module_bert_crf import CCKS2020_NER_BERT

if __name__ == '__main__':
    DEFAULT_CONFIG['model_name'] = 'bert_bilstm_crf'
    DEFAULT_CONFIG['experiment_name'] = 'bert_bilstm_crf-split'
    DEFAULT_CONFIG['pred_info_path'] = DEFAULT_CONFIG['pred_info_path'].format(DEFAULT_CONFIG['model_name'],
                                                                               DEFAULT_CONFIG['experiment_name'])
    ccks2020_ner_bert = CCKS2020_NER_BERT()
    ccks2020_ner_bert.train()
    # ccks2020_ner_bert.predict()
