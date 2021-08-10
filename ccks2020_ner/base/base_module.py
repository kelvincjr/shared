#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_module
# @Author   : 研哥哥
# @Time     : 2020/6/12 15:33


from abc import ABCMeta, abstractclassmethod


class BaseModule(metaclass=ABCMeta):
    @abstractclassmethod
    def train(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def evaluate(cls, *args, **kwargs):
        pass

    @abstractclassmethod
    def predict(cls, *args, **kwargs):
        pass
