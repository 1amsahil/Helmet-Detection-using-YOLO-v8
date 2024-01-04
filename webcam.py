from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2

model =YOLO("Helmet.pt")
result =model.predict(source ="0", show =True, conf= 0.5)
print(result)
