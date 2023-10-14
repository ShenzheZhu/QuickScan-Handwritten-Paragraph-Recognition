#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 20:22
# @Author  : FywOo02
# @FileName: test.py
# @Software: PyCharm
import os
current_directory = os.path.dirname(__file__)
print(current_directory)
# 构建相对路径，从当前脚本所在的目录到图像的路径
relative_path = os.path.join( "..", "handwritten_to_digit", "input_sentences")
print(relative_path)
# 获取图像的绝对路径
image_path = os.path.abspath(os.path.join(current_directory, relative_path))
print(image_path)