import sys
import os

from Model_development.handwritten_to_digit.Inference import startPrediction
from Model_development.line_segmentation.line_segementation_model import start_line_seg
import time


if __name__ == "__main__":
    start_line_seg()
    startPrediction()
