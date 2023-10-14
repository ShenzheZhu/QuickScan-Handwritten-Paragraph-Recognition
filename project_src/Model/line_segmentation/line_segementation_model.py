#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 20:34
# @Author  : FywOo02
# @FileName: line_segementation_model.py
# @Software: PyCharm

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil


def read_file():
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join("..", "..", "connectivity", "src_image",
                                 "sample.jpg")
    image_path = os.path.abspath(os.path.join(current_directory, relative_path))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def resizing(img):
    # print(img.shape)
    height, width, channel = img.shape

    if width > 888:
        ar = width / height
        new_w = 888
        new_h = int(new_w / ar)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # plt.imshow(img)
    # plt.show()
    return img


def img_enhance(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh, cmap='gray')
    # plt.show()
    return thresh


def img_dilation(img):
    kernel = np.ones((3, 100), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    # plt.imshow(dilated, cmap='gray')
    # plt.show()
    return dilated


def find_contours(img):
    (contours, heirarchy) = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours,
                                   key=lambda ctr: cv2.boundingRect(ctr)[
                                       1])  # (x, y, w, h)
    return sorted_contours_lines


def segment(img, contours):
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 100, 250), 2)

    plt.imshow(img)
    plt.show()


def img_segment(img, sorted_contours_lines):
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join("..", "handwritten_to_digit",
                                 "input_sentences")
    path = os.path.abspath(os.path.join(current_directory, relative_path))

    shutil.rmtree(path)
    os.mkdir(path)

    for idx, ctr in enumerate(sorted_contours_lines):
        x, y, w, h = cv2.boundingRect(ctr)
        if h > 20:
            roi = img[y:y + h, x:x + w]  # Extract the ROI from the original image
            # Save the ROI as a separate image
            cv2.imwrite(os.path.join(path, f'roi_{idx}.jpg'), roi)

            # Optionally, you can also display the ROI
            cv2.imshow(f'ROI {idx}', roi)

    # Wait for a key press and close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = read_file()
    img_resized = resizing(img)
    img_enhanced = img_enhance(img_resized)
    img_dilation = img_dilation(img_enhanced)
    contours = find_contours(img_dilation)
    # ls.segment(img_resized,contours)
    img_segment(img_resized, contours)
