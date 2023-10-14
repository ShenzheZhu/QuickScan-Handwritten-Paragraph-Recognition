#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 20:22
# @Author  : FywOo02
# @FileName: test.py
# @Software: PyCharm

import tensorflow as tf
# 查看gpu和cpu的数量
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
