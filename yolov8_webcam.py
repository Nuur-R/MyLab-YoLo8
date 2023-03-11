import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# program python untuk mengambil gambar dari webcam dan keyboard q untuk menghentikan program
cap = cv2.VideoCapture(0)
model = YOLO("models/yolov8s.pt")
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

while(True):
    ret, frame = cap.read()
    
    result = model(frame)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels
    )
    cv2.imshow('YoloV8',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break