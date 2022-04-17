import os
import re
from os.path import isfile, join

import cv2
import numpy as np


def divide_video_into_frames():
    capturedVideo = cv2.VideoCapture('input-fixed.mp4')
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
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def load_sort_frames():
    col_frames = os.listdir('frames/')
    col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
    col_images = []
    for i in col_frames:
        img = cv2.imread('frames/' + i)
        col_images.append(img)

    return col_images


def masking_lane_detection_writing(col_images):
    mask = np.zeros_like(col_images[0][:, :, 0])
    # specify coordinates of the polygon
    polygon = np.array([[100, 400], [300, 225], [500, 225], [700, 400]])
    # fill polygon with ones
    cv2.fillConvexPoly(mask, polygon, 1)
    cnt = 0
    drift_count = 0
    car_pos = (400, 360)
    pixel_to_meter = 0.0002645833
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 255)
    font_size = 0.5
    for img in col_images:
        masked = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=mask)
        ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=20)
        image_copy = img.copy()
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                else:
                    slope = 0

                cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)

                if slope > 0.95 or slope < -0.95:
                    drift_count += 1
                else:
                    drift_count = 0

                if drift_count >= 5:
                    alert_driver(image_copy)

            cv2.putText(image_copy, 'car center', (370, 340), font, font_size, font_color, 1)
            cv2.circle(image_copy, car_pos, radius=10, color=(0, 255, 0), thickness=-1)
            cv2.imwrite('detected/' + str(cnt) + '.png', image_copy)
        except TypeError:
            cv2.imwrite('detected/' + str(cnt) + '.png', img)
        cnt += 1


def alert_driver(img_cpy):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 255)
    font_size = 0.5

    cv2.putText(img_cpy,
                'Drift Alert !!!',
                (350, 320),
                color=font_color,
                fontFace=font,
                fontScale=font_size,
                thickness=1)


def video_from_detected():
    path_in = 'detected/'
    path_out = 'lane_detected.mp4'
    fps = 30.0
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    frame_list = []

    for i in range(len(files)):
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
