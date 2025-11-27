import os
import numpy as np
# Now import YOLO
from ultralytics import YOLO

import cv2

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    # Draw boxes
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Real-Time", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
