#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_model
# @Author   : 研哥哥
# @Time     : 2020/6/12 15:32

import os

import torch
import torch.nn as nn
from utils.log import logger


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.save_path = args.save_model_path

    def load(self, path=None):
        path = path if path else self.save_path
        # map_location = None if torch.cuda.is_available() else 'cpu'
        map_location = 'cpu'
        model_path = os.path.join(path, 'model.pkl')
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logger.info('loadding model from {}'.format(model_path))

    def save(self, path=None):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, 'model.pkl')
        torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))
