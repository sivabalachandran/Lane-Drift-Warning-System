import os
import re
from os.path import isfile, join

import cv2


class Processor:
    def __init__(self):
        self.input_video = 'input.mp4'
        self.lane_color = (48, 111, 178)
        self.lane_thickness = 3
        self.car_center_text_position = (370, 340)
        self.car_pos = (400, 360)
        self.font = cv2.FONT_ITALIC
        self.font_color = (7, 120, 242)
        self.font_size = 0.5
        self.alert = self.rescale_convert_frame(cv2.imread('alert-icon.png'), scale=0.10)
        self.collision = self.rescale_convert_frame(cv2.imread('collision.jpeg'), scale=0.10)

    def divide_video_into_frames(self):
        captured_video = cv2.VideoCapture(self.input_video)
        count = 0
        while captured_video.isOpened() and count <= 10000:
            success, frame = captured_video.read()
            if success:
                frame_scaled = self.rescale_convert_frame(frame, 0.5)
                cv2.imwrite("frames/frame%d.png" % count, frame_scaled)
            count += 1
        print("Video framing done")

    def rescale_convert_frame(self, frame, scale=0.75):
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)
        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    def load_sort_frames(self):
        col_frames = os.listdir('frames/')
        col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
        col_images = []
        for i in col_frames:
            img = cv2.imread('frames/' + i)
            col_images.append(img)

        return col_images

    def video_from_detected(self):
        print("Video stitching from the processed frames started")
        path_in = 'detected/'
        path_out = 'output.mp4'
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

    def write_lane(self, point1, point2, img):
        cv2.line(img, point1, point2, self.lane_color, self.lane_thickness)

    def car_center(self, img):
        cv2.putText(img, 'car center', self.car_center_text_position, self.font, self.font_size, self.font_color, 1)
        cv2.circle(img, self.car_pos, radius=10, color=self.font_color, thickness=-1)

    def alert_driver(self, img):
        rows, cols, channels = self.alert.shape
        img[300:300 + rows, 390:390 + cols, :] = self.alert

    def alert_driver_collision(self, img):
        rows, cols, channels = self.collision.shape
        img[270:270 + rows, 390:390 + cols, :] = self.collision
