import os
import re
from os.path import isfile, join

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm_notebook


def divide_video_into_frames():
    vidcap = cv2.VideoCapture('video-trim.mp4')
    count = 0
    while vidcap.isOpened() and count <= 500:
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
        print(i)
        img = cv2.imread('frames/' + i)
        col_images.append(img)

    idx = 0
    plt.figure(figsize=(10, 10))
    plt.imshow(col_images[idx][:, :, 0], cmap="gray")
    plt.show()
    return col_images


def masking_lane_detection_writing(col_images):
    idx = 0
    stencil = np.zeros_like(col_images[idx][:, :, 0])
    # specify coordinates of the polygon
    polygon = np.array([[200, 750], [600, 450], [950, 450], [1300, 750]])
    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(stencil, cmap="gray")
    plt.show()
    cnt = 0
    for img in tqdm_notebook(col_images):
        masked = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=stencil)
        ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=200)
        dmy = img.copy()
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.imwrite('detected/' + str(cnt) + '.png', dmy)
        except TypeError:
            cv2.imwrite('detected/' + str(cnt) + '.png', img)
        cnt += 1


def video_from_detected():
    pathIn = 'detected/'
    pathOut = 'lane_detected.mp4'
    fps = 30.0
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    frame_list = []

    for i in tqdm_notebook(range(len(files))):
        filename = pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_list.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_list)):
        # writing to a image array
        out.write(frame_list[i])

    out.release()


if __name__ == "__main__":
    divide_video_into_frames()
    col_images = load_sort_frames()
    masking_lane_detection_writing(col_images)
    video_from_detected()
