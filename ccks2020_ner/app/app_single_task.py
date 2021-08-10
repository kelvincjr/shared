#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : app_single_task
# @Author   : LiuYan
# @Time     : 2021/5/20 2:18

from base.base_app import *
from runner.runner_pred import *

single_task = Blueprint('/transfer_ner/single_task', __name__)


@single_task.route('/test', methods=('GET', 'POST'))
def test():
    logger.info('test -> transfer_ner -> single_task success!')
    return 'test -> transfer_ner -> single_task success!'


@single_task.route('/pred', methods=['POST'])
def pred():
    """
    -> data:
    :return:
    """
    data = request.get_json()
    content = data['content'] if 'content' in data else None

    result_dict = ccks2020_ner.pred(content=content, type='ST')
    # logger.info(result_dict)
    app.logger.info(result_dict)

    return json.dumps(result_dict, ensure_ascii=False)
