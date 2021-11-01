#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 工具代码合集
# 注：最好写绝对路径，否则可能出现无法预料的错误。
# 科学空间：https://kexue.fm

import numpy as np
from rouge import Rouge
import os, sys
import jieba
from bert4keras.snippets import open

# 自定义词典
user_dict_path = './datasets/user_dict.txt'
user_dict_path_2 = './datasets/user_dict_2.txt'
jieba.load_userdict(user_dict_path)
jieba.initialize()

# 设置递归深度
sys.setrecursionlimit(1000000)

# 标注数据
#data_json = './datasets/train.json'
data_json = './bdci_datasets/train_dataset.csv'

# 保存权重的文件夹
if not os.path.exists('weights'):
    os.mkdir('weights')

# bert配置
#config_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
#dict_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

config_path = '/kaggle/input/bert-roberta/bert_config.json'
checkpoint_path = '/kaggle/input/bert-roberta/bert_model.ckpt'
dict_path = '/kaggle/input/bert-roberta/vocab.txt'

# nezha配置
nezha_config_path = '/kaggle/input/nezha_base/bert_config.json'
nezha_checkpoint_path = '/kaggle/input/nezha_base/bert_model.ckpt'
nezha_dict_path = '/kaggle/input/nezha_base/vocab.txt'

# 将数据划分N份，一份作为验证集
num_folds = 1 #15

# 指标名
metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

# 计算rouge用
rouge = Rouge()


def load_user_dict(filename):
    """加载用户词典
    """
    user_dict = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            w = l.split()[0]
            user_dict.append(w)
    return user_dict


def data_split(data, fold, num_folds, mode):
    """划分训练集和验证集
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]

    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D


def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l
    """
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target, unit='word'):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics


def compute_main_metric(source, target, unit='word'):
    """计算主要metric
    """
    return compute_metrics(source, target, unit)['main']

def longest_common_subsequence(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]
