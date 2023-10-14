#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 20:43
# @Author  : FywOo02
# @FileName: test.py
# @Software: PyCharm

import cv2
import sys
import line_segementation_model as ls

if __name__ == "__main__":
    img = ls.read_file()
    img_resized = ls.resizing(img)
    img_enhanced = ls.img_enhance(img_resized)
    img_dilation = ls.img_dilation(img_enhanced)
    contours = ls.find_contours(img_dilation)
    # ls.segment(img_resized,contours)
    ls.img_segment(img_resized,contours)

