import os
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm_notebook


def divide_video_into_frames():
    vidcap = cv2.VideoCapture('lane_detection_input.mp4')
    count = 0
    while vidcap.isOpened() and count <= 10:
        success, image = vidcap.read()
        if success:
            cv2.imwrite("frames/frame%d.png" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def load_sort_frames():
    col_frames = os.listdir('frames/')
    col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
    col_images = []
    for i in tqdm_notebook(col_frames):
        img = cv2.imread('frames/' + i)
        col_images.append(img)

    idx = 0

    # plot frame
    plt.figure(figsize=(10, 10))
    plt.imshow(col_images[idx][:, :, 0], cmap="gray")
    plt.show()
    return col_images


def masking(col_images):
    idx = 0
    stencil = np.zeros_like(col_images[idx][:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[500, 1000], [700, 600], [1500, 600], [1750, 1000]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(stencil, cmap="gray")
    plt.show()
    img = cv2.bitwise_and(col_images[idx][:, :, 0], col_images[idx][:, :, 0], mask=stencil)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    plt.show()

    ret, thresholded_image = cv2.threshold(img, 145, 200, cv2.THRESH_BINARY)

    # plot image
    plt.figure(figsize=(10, 10))
    plt.imshow(thresholded_image, cmap="gray")
    plt.show()

    lines = cv2.HoughLinesP(thresholded_image, 1, np.pi / 180, 30, maxLineGap=200)

    # create a copy of the original frame
    dmy = col_images[idx][:, :, 0].copy()

    # draw Hough lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # plot frame
    plt.figure(figsize=(10, 10))
    plt.imshow(dmy, cmap="gray")
    plt.show()


if __name__ == "__main__":
    #divide_video_into_frames()
    col_images = load_sort_frames()
    masking(col_images)
