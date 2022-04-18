import cv2
import torch


class Transformer:
    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.conf = 0.5
        model.classes = [0, 2]
        self.model = model

    def yolo_transform(self, img):
        result = self.model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result.render()
        return result.imgs[0]

    def detect_for_collision(self, img):
        img_cpy = img.copy()
        cropped_image = img_cpy[100:400, 300:500]
        df = self.model(cropped_image, size=70).pandas().xyxy[0]
        df = df[df['confidence'] > 0.5]
        return not df.empty
