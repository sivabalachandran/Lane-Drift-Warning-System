import cv2
import torch


class Transformer:
    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.conf = 0.5
        self.model = model

    def yolo_transform(self, img):
        result = self.model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result.render()
        return result.imgs[0]
