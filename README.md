# Lane Drift Warning System

## Frameworks and Language
- Python 3.9
- OpenCV
- Yolov5
- PyTorch

## Code Structure
- Install all the requirements using ` pip install -r requirements.txt `
- `lane_detection.py` is the main file that acts as starting point and does all the processing. Run it using `python3 lane-detecion.py`
- `FrameProcessing.py` does all video and image frame related processing.
- `YoloTransformation.py` does the job of object detection.
- `cleanup.sh` cleans up the workspace.

To run the program, download this [video](https://kennesawedu-my.sharepoint.com/:v:/g/personal/sbalacha_students_kennesaw_edu/EbXox33un-1DsQ6BmkJX07oBb_NoMPXGccSXLSP3OvHYUw?email=skhandav%40students.kennesaw.edu&e=l60OQx) and use it as input. 
Note: The programe is limited to process only 10,000 frames. Yolo detection does take long time and consume resources.

## Objective
The main objective of the project is to detect lanes and warn driver when the vehicle drifts. As an added bonus I have added Yolo object detection that detects cars and people. Collision alert is in pipeline.

## Process Steps
- Split video into frames.
- For each frame do the following pre-processing 
   - Scale it down for faster processing. The image frame can be converted to grey scale as well, but for better visualization I have kept in 3 channel colors.
   - Yolo object detection. I have used a pre-trained model here in this project
   - Create area of interest (AOI) polygon.
   - Apply the mask on the frame using the polygon created above. This helps to avoid noise and detect lanes better.
   - Apply thresholding
   - Hough transformation to create right and left lanes.
   - Calculate slope of the lanes to detect drifts. 
   - Alert when the drift count meets a threshold.
- Stictch frames to make a outout video.
   
