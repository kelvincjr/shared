#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_config
# @Author   : 研哥哥
# @Time     : 2020/6/12 15:32

DEFAULT_CONFIG = {

}


class BaseConfig(object):
    def __init__(self):
        pass

    @staticmethod
    def load(path=None):
        """
        loading config
        :return:
        """
        pass

    def save(self, path=None):
        """
        save config
        :param path:
        :return:
        """
        pass
