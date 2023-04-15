import os
from ultralytics import YOLO
import cv2
import math
import numpy as np

VIDEO_DIR = os.path.join(".", "videos")
VIDEO_PATH = os.path.join(VIDEO_DIR, "mtid_video_test.mp4")
# Output file path (change name if necessary)
ANNOT_VIDEO_PATH = os.path.join(VIDEO_DIR, "mitd_video_annot3.mp4")
MODEL_FILE_PATH = os.path.join(".", "weights", "best-230404.pt")
FPS = 30
SCORE_THRESHOLD = 0.6
PROXIMITY_THRESHOLD = 30 # pixels
OFFSET = 7 # pixels

class Vehicle:
    def __init__(self, id, class_id, x1, y1, x2, y2):
        self.id = id
        self.class_id = class_id
        self.pt1 = (int(x1), int(y1))
        self.pt2 = (int(x2), int(y2))
        self.center = (int((x1 + x2)/2), int((y1 + y2)/2))
    
    def Update(self, class_id, x1, y1, x2, y2):
        self.class_id = class_id
        self.pt1 = (int(x1), int(y1))
        self.pt2 = (int(x2), int(y2))
        self.center = (int((x1 + x2)/2), int((y1 + y2)/2))

class VehicleCounter:

    def __init__(self):
        self.counterID = 0
        self.VehiclesLastFrame = []

    def Add(self, NewDetections):
        for x1, y1, x2, y2, score, class_id in NewDetections:
            self.VehiclesLastFrame.append(Vehicle(self.counterID, class_id, x1, y1, x2, y2))
            self.counterID += 1

    def ProtectedAdd(self, NewDetections):
        for x1, y1, x2, y2, score, class_id in NewDetections:
            if score > SCORE_THRESHOLD:
                self.VehiclesLastFrame.append(Vehicle(self.counterID, class_id, x1, y1, x2, y2))
                self.counterID += 1
    
    def Attendance(self, vehicle_ids):
        s = []
        self.VehiclesLastFrame = [vehicle for vehicle in self.VehiclesLastFrame if vehicle.id in vehicle_ids]
        for vehicle1 in self.VehiclesLastFrame:
            duplicate = False
            for vehicle2 in self.VehiclesLastFrame:
                if math.hypot(vehicle1.center[0] - vehicle2.center[0], vehicle1.center[1] - vehicle2.center[1]) < 10 and vehicle1.id != vehicle2.id:
                    duplicate = True
            if not duplicate:
                s.append(vehicle1)
        self.VehiclesLastFrame = s
        # Simple version, does not clean the memory -> self.VehiclesLastFrame = [vehicle for vehicle in self.VehiclesLastFrame if vehicle.id in vehicle_ids]
    
    def Update(self, detections):
        VehiclesPresent = []
        NewDetections = detections.copy()

        for vehicle in self.VehiclesLastFrame:
            vehiclePresent = False

            for detection in detections:
                x1, y1, x2, y2, score, class_id = detection
                if score > SCORE_THRESHOLD:
                    cx = (x1 + x2)/2
                    cy = (y1 + y2)/2
                    if math.hypot(vehicle.center[0] - cx, vehicle.center[1] - cy) < PROXIMITY_THRESHOLD:
                        vehicle.Update(class_id, x1, y1, x2, y2)
                        vehiclePresent = True
                        if detection in NewDetections:
                            NewDetections.remove(detection)

            if vehiclePresent:
                VehiclesPresent.append(vehicle.id)
        
        self.Attendance(VehiclesPresent)
        self.Add(NewDetections)


if __name__ == "__main__":

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    VideoCapture = cv2.VideoCapture(VIDEO_PATH)
    model = YOLO(MODEL_FILE_PATH)
    classes = {0: "car", 1: "bicycle", 2: "bus", 3: "lorry"} 

    ret, frame = VideoCapture.read()
    vc = VehicleCounter()
    vc.ProtectedAdd(model(frame)[0].boxes.data.tolist())
    height, width, channels = frame.shape

    VideoWriter = cv2.VideoWriter(ANNOT_VIDEO_PATH, fourcc, 24, (width, height))

    ret, frame = VideoCapture.read()
    while True:

        if ret:
            # list of ultralytics.yolo.engine.results.Results
            detections = model(frame)[0].boxes.data.tolist()

            vc.Update(detections)

            for vehicle in vc.VehiclesLastFrame:

                cv2.rectangle(
                    img=frame, 
                    pt1=vehicle.pt1, 
                    pt2=vehicle.pt2, 
                    color=(20, 255, 57), 
                    thickness=2
                )
                # add text to image
                cv2.putText(
                    img=frame, 
                    text=classes.get(vehicle.class_id), 
                    org=(vehicle.pt1[0], vehicle.pt1[1]-OFFSET), 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=1,
                    color=(20, 255, 57), # bgr 
                    thickness=3
                )
                cv2.circle(
                    img=frame,
                    center=(vehicle.center[0], vehicle.center[1]),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1
                )
                cv2.putText(
                    img=frame,
                    text=str(vehicle.id),
                    org=(vehicle.center[0], vehicle.center[1]-OFFSET),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, 
                    color=(0, 0, 255), 
                    thickness=2
                )

            cv2.imshow("Frame", frame)
            VideoWriter.write(frame)

            # If the "esc" key is pressed
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        else:
            break

        ret, frame = VideoCapture.read()

    VideoCapture.release()
    VideoWriter.release()
    cv2.destroyAllWindows()


