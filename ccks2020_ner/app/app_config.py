#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : app_config
# @Author   : LiuYan
# @Time     : 2021/4/22 10:51

import os
import multiprocessing

from pathlib import Path

bind = '0.0.0.0:7000'               # 绑定ip和端口号
backlog = 512                       # 监听队列
# chdir = '/home/zzsn/liuyan/bin'     # gunicorn要切换到的目的工作目录
timeout = 30                        # 超时
worker_class = 'gevent'             # 使用gevent模式，还可以使用sync 模式，默认的是sync模式

# workers = multiprocessing.cpu_count() * 2 + 1     # 进程数 12 * 2 + 1 = 25
workers = 1                                         # 低资源 3.8G 待测试如何减小
threads = 2                                         # 指定每个进程开启的线程数
loglevel = 'info'                   # 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'  # 设置gunicorn访问日志格式，错误日志无法设置

"""
其每个选项的含义如下：
h          remote address
l          '-'
u          currently '-', may be user name in future releases
t          date of the request
r          status line (e.g. ``GET / HTTP/1.1``)
s          status
b          response length or '-'
f          referer
a          user agent
T          request time in seconds
D          request time in microseconds
L          request time in decimal seconds
p          process ID
"""
_tmp_path = os.path.dirname(os.path.abspath(__file__))
_tmp_path = os.path.join(_tmp_path, 'log')
Path(_tmp_path).mkdir(parents=True, exist_ok=True)
accesslog = os.path.join(_tmp_path, 'gunicorn_access.log')      # 访问日志文件
errorlog = os.path.join(_tmp_path, 'gunicorn_info.log')         # 错误日志文件

# gunicorn -c app_config.py app_run:app -D --daemon
