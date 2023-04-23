import os
from ultralytics import YOLO
import cv2
import math
import numpy as np
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.detection import Detection
from deep_sort.application_util import preprocessing
from deep_sort.tools import generate_detections as gdet
import torch

VIDEO_DIR = os.path.join(".", "videos")
VIDEO_PATH = os.path.join(VIDEO_DIR, "mtidVideo_test1.mp4")
# Output file path (change name if necessary)
ANNOT_VIDEO_PATH = os.path.join(VIDEO_DIR, "mitd_video_annot3.mp4")
MODEL_FILE_PATH = os.path.join(".", "weights", "best-230404.pt")
FPS = 30
CONFIDENCE_THRESHOLD = 0.5
PROXIMITY_THRESHOLD = 30 # pixels
OFFSET = 7 # pixels



if __name__ == "__main__":

    metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.2, budget=None)
    tracker = Tracker(metric, max_age=30, n_init=2)
    encoder = gdet.create_box_encoder("./mars-small128.pb", batch_size=1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    VideoCapture = cv2.VideoCapture(VIDEO_PATH)
    model = YOLO(MODEL_FILE_PATH)
    classes = {0: "car", 1: "bicycle", 2: "bus", 3: "lorry"} 

    ret, frame = VideoCapture.read()
    height, width, channels = frame.shape

    # VideoWriter = cv2.VideoWriter(ANNOT_VIDEO_PATH, fourcc, 24, (width, height))

    ret, frame = VideoCapture.read()
    while True:

        if ret:
            # list of ultralytics.yolo.engine.results.Results
            matrix = model(frame)[0].boxes.data.detach().cpu().numpy()

            mask = matrix[:, 4] >= CONFIDENCE_THRESHOLD
            matrix = matrix[mask]
            
            matrix[:, 2:4] = matrix[:, 2:4] - matrix[:, :2]
            indices = preprocessing.non_max_suppression(matrix[:, :4], max_bbox_overlap=0.8, scores=matrix[:, -1])
            matrix = matrix[indices]
            # featureVector shape: [12, 128]
            featureVector = encoder(frame, matrix[:, :4])

            detections = [Detection(m[:4], m[4], f_vector) for m, f_vector in zip(matrix, featureVector)]

            tracker.predict()
            tracker.update(detections)

            tracks = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr().astype(int)

                id = track.track_id

                tracks.append((bbox, id))

            for bbox, id in tracks:
                cv2.rectangle(
                    img=frame, 
                    pt1=bbox[:2], 
                    pt2=bbox[2:4], 
                    color=(20, 255, 57), 
                    thickness=2
                )
                # add text to image
                cv2.putText(
                    img=frame, 
                    text=str(id), 
                    org=(bbox[0], bbox[1] - OFFSET), 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=1,
                    color=(20, 255, 57), # bgr 
                    thickness=3
                )

            cv2.imshow("Frame", frame)
            # # VideoWriter.write(frame)

            # If the "esc" key is pressed
            key = cv2.waitKey(0)
            if key == 27:
                break
        
        else:
            break

        ret, frame = VideoCapture.read()

    VideoCapture.release()
    # VideoWriter.release()
    cv2.destroyAllWindows()


