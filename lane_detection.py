import cv2
import numpy as np

from FrameProcessing import Processor
from YoloTransformation import Transformer


def masking_lane_detection_writing(col_images):
    print("Masking and Lane detection started")
    mask = np.zeros_like(col_images[0][:, :, 0])
    # specify coordinates of the polygon
    polygon = np.array([[100, 400], [300, 225], [500, 225], [700, 400]])
    # fill polygon with ones
    cv2.fillConvexPoly(mask, polygon, 1)
    cnt = 0
    drift_count = 0
    yolo_transform_instance = Transformer()
    for img in col_images:
        # Masking, Thresholding and Lane created using Hough transform
        masked = cv2.bitwise_and(img[:, :, 0], img[:, :, 0], mask=mask)
        ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, maxLineGap=20)

        image_copy = img.copy()
        # Object detection using YOLO.
        image_copy = yolo_transform_instance.yolo_transform(image_copy)
        # detect for collision
        collision = yolo_transform_instance.detect_for_collision(image_copy)
        if collision:
            processor.alert_driver_collision(image_copy)
        # Attach car center
        processor.car_center(image_copy)
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                processor.write_lane((x1, y1), (x2, y2), image_copy)

                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                else:
                    slope = 0

                if slope > 0.95 or slope < -0.95:
                    drift_count += 1
                else:
                    drift_count = 0

                if drift_count >= 5:
                    processor.alert_driver(image_copy)

            cv2.imwrite('detected/' + str(cnt) + '.png', image_copy)
        except TypeError:
            cv2.imwrite('detected/' + str(cnt) + '.png', img)
        cnt += 1


if __name__ == "__main__":
    processor = Processor()
    processor.divide_video_into_frames()
    col_images = processor.load_sort_frames()
    masking_lane_detection_writing(col_images)
    processor.video_from_detected()
