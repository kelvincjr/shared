#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : base_app
# @Author   : LiuYan
# @Time     : 2021/4/21 9:30

import json

from flask import Flask, Blueprint, request

from utils.log import logger

app = Flask(__name__)
