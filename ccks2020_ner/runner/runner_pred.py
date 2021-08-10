#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : runner_pred
# @Author   : LiuYan
# @Time     : 2021/5/20 2:57

from config.config import DEFAULT_CONFIG
from module.module import CCKS2020_NER

DEFAULT_CONFIG['model_name'] = 'transformer_bilstm_crf'
DEFAULT_CONFIG['experiment_name'] = '10_07_1745-b-t_d-dae_1.2-dice_0.01-result_valid'
DEFAULT_CONFIG['save_model_path'] = DEFAULT_CONFIG['save_model_path'].format(DEFAULT_CONFIG['model_name'])
DEFAULT_CONFIG['result_path'] = DEFAULT_CONFIG['result_path'].format(DEFAULT_CONFIG['model_name'])
ccks2020_ner = CCKS2020_NER()
ccks2020_ner.ready()

if __name__ == '__main__':
    result_dict = ccks2020_ner.pred(
        content='患者于1年前因“体检发现左肺占位7天”入我院。入院后完善相关检查，排除手术禁忌症后，于2014年12月29日全麻下行“左肺上叶癌根治术”，病检示：“（左肺上叶结节）腺癌，部分为贴壁生长型，冰冻取材剩余肺组织、支气管残端于镜下未见癌组织。支气管旁淋巴结（0/3）、另送4组（0/3）、5组（0/1）、10组（0/1）淋巴结于镜下未见癌组织转移。术后恢复好，切口愈合好，予安排出院。之后行两次复查，无复发及转移，给免疫、对症治疗，无不适反应。于2015-02-27、2015-5-4、2015-8-26cik细胞治疗3疗程，无特殊不适。行今为进一步治疗来我院就诊，门诊以“左肺上叶腺癌”收入院。自发病以来，病人精神状态良好，体力情况良好，食欲食量正常，睡眠情况良好，体重无明显变化，大便正常，小便正常。', type='ST')
    pass
