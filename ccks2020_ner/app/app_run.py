#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : app_run
# @Author   : LiuYan
# @Time     : 2021/4/20 20:11

import sys
import logging

sys.path.append('../')
from base.base_app import *
from app.app_single_task import single_task
from app.app_multi_task import multi_task
from app.app_ours import ours

HOST = '0.0.0.0'
PORT = 7000
DEBUG = False

# classification
app.register_blueprint(single_task, url_prefix='/transfer_ner/single_task')
app.register_blueprint(multi_task, url_prefix='/transfer_ner/multi_task')
app.register_blueprint(ours, url_prefix='/transfer_ner/ours')


@app.route('/')
def hello_world():
    app.logger.info('Hello World!')
    return 'Hello World!'


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
