#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : app_ours
# @Author   : LiuYan
# @Time     : 2021/5/20 2:56

from base.base_app import *
from runner.runner_pred import *

ours = Blueprint('/transfer_ner/ours', __name__)


@ours.route('/test', methods=('GET', 'POST'))
def test():
    logger.info('test -> transfer_ner -> ours success!')
    return 'test -> transfer_ner -> ours success!'


@ours.route('/pred', methods=['POST'])
def pred():
    """
    -> data:
    :return:
    """
    data = request.get_json()
    content = data['content'] if 'content' in data else None

    result_dict = ccks2020_ner.pred(content=content, type='OURS')
    # logger.info(result_dict)
    app.logger.info(result_dict)

    return json.dumps(result_dict, ensure_ascii=False)
