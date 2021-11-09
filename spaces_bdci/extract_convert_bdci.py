#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 抽取式：数据转换
# 科学空间：https://kexue.fm

import os
import json
import heapq
import numpy as np
from tqdm import tqdm
from bert4keras.snippets import open
from bert4keras.snippets import text_segmentate
from bert4keras.snippets import parallel_apply
from snippets import *

# 初始化
maxlen = 128 #256


def text_split(text, limited=True):
    """将长句按照标点分割为多个子句。
    """
    #texts = text_segmentate(text, 1, u'\n。；：，')
    texts = text_segmentate(text, 1, u'\n【')
    if limited:
        texts = texts[-maxlen:]
    texts = texts[1:]
    return texts

def summary_split(text, limited=True):
    """将长句按照标点分割为多个子句。
    """
    texts = text_segmentate(text, 1, u'\n。；：，')
    #texts = text_segmentate(text, 1, u'\n【')
    if limited:
        texts = texts[-maxlen:]
    return texts


def extract_matching(texts, summaries, start_i=0, start_j=0):
    """在texts中找若干句子，使得它们连起来与summaries尽可能相似
    算法：texts和summaries都分句，然后找出summaries最长的句子，在texts
          中找与之最相似的句子作为匹配，剩下部分递归执行。
    """
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])
    #j = np.argmax([compute_main_metric(t, summaries[i], 'char') for t in texts])
    metric_array = np.array([compute_main_metric(t, summaries[i], 'char') for t in texts])
    j_top3 = heapq.nlargest(2, range(len(metric_array)), metric_array.take)
    j = j_top3[0]
    #========== debug ============
    '''
    text_id = 0
    for t in texts:
        sim = compute_main_metric(t, summaries[i], 'char')
        print('({},{}), sim: [{}], summary: [{}], text: [{}]'.format(i, text_id, sim, summaries[i], t))
        text_id += 1
    print('**********best, ({},{}), summary: [{}], text: [{}]'.format(i, j, summaries[i], texts[j]))
    '''
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    pairs = [(start_i + i, start_j + k) for k in j_top3]
    #return lm + [(start_i + i, start_j + j)] + rm
    return lm + pairs + rm


def extract_flow(inputs):
    """单个样本的构建流（给parallel_apply用）
    """
    text, summary = inputs
    texts = text_split(text, True)  # 取后maxlen句
    summaries = summary_split(summary, False)
    mapping = extract_matching(texts, summaries)
    labels = sorted(set([i[1] for i in mapping]))
    labels = [int(label) for label in labels]
    pred_summary = ''.join([texts[i] for i in labels])
    metric = compute_main_metric(pred_summary, summary)
    return texts, labels, summary, metric

'''
def load_data(filename):
    """加载数据
    返回：[(text, summary)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            text = '\n'.join([d['sentence'] for d in l['text']])
            D.append((text, l['summary']))
    return D
'''
def load_data(filename):
    """加载数据
    返回：[(text, summary)]
    """
    D = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split('|')
            id_ = parts[0]
            content = parts[1]
            abstract = parts[2]
            content_len = len(content)
            abstract_len = len(abstract)
            D.append((content, abstract))
    return D


def convert(data):
    """分句，并转换为抽取式摘要
    """
    D = parallel_apply(
        func=extract_flow,
        iterable=tqdm(data, desc=u'转换数据'),
        workers=100,
        max_queue_size=200
    )
    total_metric = sum([d[3] for d in D])
    D = [d[:3] for d in D]
    print(u'抽取结果的平均指标: %s' % (total_metric / len(D)))
    return D


if __name__ == '__main__':
    '''
    source = '客户】嗯，99还用不了那么多，【'
    target = '望我方为其退还5元日包费用，'
    source = source.replace(' ', '').replace('坐席】', '').replace('客户】', '').replace('【', '')
    import re
    zz = re.compile(u'[\n。；：，]')
    source = zz.sub('', source)
    target = target.replace(' ', '').replace('坐席】', '').replace('客户】', '').replace('【', '')
    target = zz.sub('', target)
    print('source: {}, target: {}'.format(source,target))
    metrics = compute_rouge(source, target, unit='char')
    print('metrics: ', metrics)
    print('metrics main: ', compute_main_metric(source, target, 'char'))
    import sys
    sys.exit(0)
    '''

    data_random_order_json = data_json[:-4] + '_random_order.json'
    data_extract_json = data_json[:-4] + '_extract.json'

    data = load_data(data_json)

    #========== debug ============
    '''
    max_ts_num = 0
    ts_num_128 = 0
    ts_num_256 = 0
    total_ts_num = 0
    total_count = 0
    for inputs in tqdm(data, desc=u'转换数据'):
        text, summary = inputs
        texts = text_split(text, True)  # 取后maxlen句
        if len(texts) > max_ts_num:
            max_ts_num = len(texts)
        if len(texts) >= 128:
            ts_num_128 += 1
        if len(texts) >= 256:
            ts_num_265 += 1
        total_ts_num += len(texts)
        summaries = summary_split(summary, False)
        mapping = extract_matching(texts, summaries)
        labels = sorted(set([i[1] for i in mapping]))
        labels = [int(label) for label in labels]
        total_count += 1
        print('==================== text ============================')
        sent_num = 0
        for item in texts:
            lb = 0
            if sent_num in labels:
                lb = 1
            print('text: {}, label: {}'.format(item, lb))
            sent_num += 1
        print('==================== summary ============================')
        for item in summaries:
            print('summary: {}'.format(item))
        print('==================== labels {} ============================'.format(labels))
        for ind in labels:
            print('label text: {}'.format(texts[ind]))
        if total_count > 1:
            break
    print('max_ts_num: [{}], avg_ts_num: [{}], ts_num_128; [{}], ts_num_256: [{}]'.format(max_ts_num, (total_ts_num)/total_count, ts_num_128, ts_num_256))
    import sys
    sys.exit(0)
    '''
    
    data = convert(data)

    if os.path.exists(data_random_order_json):
        idxs = json.load(open(data_random_order_json))
    else:
        idxs = list(range(len(data)))
        np.random.shuffle(idxs)
        json.dump(idxs, open(data_random_order_json, 'w'))

    data = [data[i] for i in idxs]

    with open(data_extract_json, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(u'输入数据：%s' % data_json)
    print(u'数据顺序：%s' % data_random_order_json)
    print(u'输出路径：%s' % data_extract_json)
