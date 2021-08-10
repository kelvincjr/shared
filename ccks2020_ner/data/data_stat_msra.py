#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_stat_msra
# @Author   : 研哥哥
# @Time     : 2020/7/15 22:04

import matplotlib.pyplot as plt
from utils.tool import tool
from config.config_msra import DEFAULT_CONFIG

train_data = tool.load_msra_data(DEFAULT_CONFIG['train_path'])
dev_data = tool.load_msra_data(DEFAULT_CONFIG['dev_path'])
train_examples = train_data.examples
dev_examples = dev_data.examples
label_stat = {'PERSON': 0,
              'LOCATION': 0,
              'ORGANIZATION': 0}
for example in dev_examples:
    tag_list = example.tag
    for tag in tag_list:
        if tag[0] == 'B' or tag[0] == 'S':
            label_stat[tag[2:]] += 1


# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))


name_list = ['PERSON', 'LOCATION', 'ORGANIZATION']
num_list = [17610, 36617, 20584]
autolabel(plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
plt.title('MSRA_train实体分布统计')
plt.xlabel('MSRA_train实体类别')
plt.ylabel('所含实体数')
plt.show()
print(label_stat)
pass
