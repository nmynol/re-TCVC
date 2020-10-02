from util.utils import load_img
from skimage import color, feature, util
import cv2
import numpy as np
# path = './data/train/frame00000.jpg'
# target = load_img(path)
# input = color.rgb2gray(target)
# print(input.shape)
# input = feature.canny(input, sigma=1)
# print(input)
# input = (input == False)
# cv2.imshow("input", input.astype('float64'))
# cv2.waitKey(0)

# for i in range(1, 5):
#     print(i)

import os
import cv2

video_cap = cv2.VideoCapture('./import.mp4')

frame_count = 0

while True:
    ret, frame = video_cap.read()
    if ret is False:
        break
    frame_count = frame_count + 1

print(frame_count)
