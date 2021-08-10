#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : app_multi_task
# @Author   : LiuYan
# @Time     : 2021/5/20 2:55

from base.base_app import *
from runner.runner_pred import *

multi_task = Blueprint('/transfer_ner/multi_task', __name__)


@multi_task.route('/test', methods=('GET', 'POST'))
def test():
    logger.info('test -> transfer_ner -> multi_task success!')
    return 'test -> transfer_ner -> multi_task success!'


@multi_task.route('/pred', methods=['POST'])
def pred():
    """
    -> data:
    :return:
    """
    data = request.get_json()
    content = data['content'] if 'content' in data else None

    result_dict = ccks2020_ner.pred(content=content, type='MT')
    # logger.info(result_dict)
    app.logger.info(result_dict)

    return json.dumps(result_dict, ensure_ascii=False)
