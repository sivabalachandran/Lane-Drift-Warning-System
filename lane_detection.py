import os
import re
from os.path import isfile, join

import cv2
import numpy as np
from cv2 import IMREAD_GRAYSCALE
from tqdm.notebook import tqdm_notebook


def divide_video_into_frames():
    capturedVideo = cv2.VideoCapture('video-trim-longer.mp4')
    count = 0
    while capturedVideo.isOpened() and count <= 5000:
        success, frame = capturedVideo.read()
        if success:
            frame_scaled = rescale_cnvrt_Frame(frame, 0.5)
            cv2.imwrite("frames/frame%d.png" % count, frame_scaled)
        count += 1


def rescale_cnvrt_Frame(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)
    img = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    gry_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gry_img


def load_sort_frames():
    col_frames = os.listdir('frames/')
    col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
    col_images = []
    for i in tqdm_notebook(col_frames):
        img = cv2.imread('frames/' + i, IMREAD_GRAYSCALE)
        col_images.append(img)

    return col_images


def masking_lane_detection_writing(col_images):
    mask = np.zeros(col_images[0].shape[:2], dtype='uint8')
    # specify coordinates of the polygon
    polygon = np.array([[100, 400], [300, 225], [500, 225], [700, 400]])
    # fill polygon with ones
    cv2.fillConvexPoly(mask, polygon, 1)
    cnt = 0
    leftLaneCount = 0
    rightLaneCount = 0
    for img in tqdm_notebook(col_images):
        masked = cv2.bitwise_and(img, img, mask=mask)
        ret, thresh = cv2.threshold(masked, 130, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Threshold", thresh)
        # cv2.waitKey(0)
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=200)
        dmy = img.copy()
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                if slope > 0.4:
                    # print("right lane")
                    leftLaneCount = 0
                    rightLaneCount = rightLaneCount + 1
                elif slope < -0.4:
                    # print("left lane")
                    rightLaneCount = 0
                    leftLaneCount = leftLaneCount + 1
                cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                if rightLaneCount > 20 or leftLaneCount > 20:
                    print('Drift Detected')
                    cv2.putText(dmy, 'Drifting', (50, 50), color=(0, 0, 0), lineType=cv2.LINE_AA,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
                # cv2.imshow("img", dmy)
                # cv2.waitKey(0)
            cv2.imwrite('detected/' + str(cnt) + '.png', dmy)

        except TypeError:
            cv2.imwrite('detected/' + str(cnt) + '.png', img)
        cnt += 1


def video_from_detected():
    path_in = 'detected/'
    path_out = 'lane_detected.mp4'
    fps = 30.0
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    frame_list = []

    for i in tqdm_notebook(range(len(files))):
        filename = path_in + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_list.append(img)

    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_list)):
        out.write(frame_list[i])

    out.release()


if __name__ == "__main__":
    divide_video_into_frames()
    col_images = load_sort_frames()
    masking_lane_detection_writing(col_images)
    video_from_detected()
