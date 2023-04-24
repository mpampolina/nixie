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
VIDEO_PATH = os.path.join(VIDEO_DIR, "mtidVideo_test2.mp4")
# Output file path (change name if necessary)
ANNOT_VIDEO_PATH = os.path.join(VIDEO_DIR, "mitd_video_annotGOOD.mp4")
MODEL_FILE_PATH = os.path.join(".", "weights", "best-230404.pt")
FPS = 30
CONFIDENCE_THRESHOLD = 0.5
PROXIMITY_THRESHOLD = 30  # pixels
OFFSET = 7  # pixels

approaches = {0: "E", 1: "W", 2: "S", 3: "N"}

dir_map = {
    "N": {"N": "SBU", "S": "SBT", "E": "SBL", "W": "SBR"},
    "S": {"N": "NBT", "S": "NBU", "E": "NBR", "W": "NBL"},
    "E": {"N": "WBR", "S": "WBL", "E": "WBU", "W": "WBT"},
    "W": {"N": "EBL", "S": "EBR", "E": "EBT", "W": "EBU"},
}


def line_equations(input):
    """
    Return the equations of a line in the form ax + by + c = 0
    and a matrix corresponding to two points along that line.

    """
    chkpt_tlbr = np.zeros((4, 4))
    eqns = np.zeros((4, 3))
    chkpt_tlbr[:, -1] = np.radians(90 - input[:, -1])
    chkpt_tlbr[:, 0:2] = input[:, 0:2]
    chkpt_tlbr[:, 2] = (input[:, 0] + input[:, 2] * np.sin(chkpt_tlbr[:, -1])).astype(
        int
    )
    chkpt_tlbr[:, 3] = (input[:, 1] - input[:, 2] * np.cos(chkpt_tlbr[:, -1])).astype(
        int
    )
    print(chkpt_tlbr)
    eqns[:, 0] = (chkpt_tlbr[:, 3] - chkpt_tlbr[:, 1]) / (
        chkpt_tlbr[:, 2] - chkpt_tlbr[:, 0]
    )
    eqns[:, 1] = np.ones(4) * -1
    eqns[:, 2] = chkpt_tlbr[:, 1] - (eqns[:, 0] * chkpt_tlbr[:, 0])
    return eqns, chkpt_tlbr


def pt_to_line_distance(pt, eqns):
    return np.abs(eqns[:, 0] * pt[0] + eqns[:, 1] * pt[1] + eqns[:, 2]) / np.sqrt(
        np.power(eqns[:, 0], 2) + np.power(eqns[:, 1], 2)
    )


def checkpoint_endpoints(eqns, res=(1920, 1080)):
    bounds = [0, res[1], 0, res[1], 0, res[0], 0, res[0]]
    left = eqns[:, 0] * 0 + eqns[:, 2]
    right = eqns[:, 0] * res[0] + eqns[:, 2]
    top = (eqns[:, 1] * 0 + eqns[:, 2]) / (-1 * eqns[:, 0])
    bottom = (eqns[:, 1] * res[1] + eqns[:, 2]) / (-1 * eqns[:, 0])
    mask_arr = np.array(
        [
            left <= bounds[0],
            left >= bounds[1],
            right <= bounds[2],
            right >= bounds[3],
            top <= bounds[4],
            top >= bounds[5],
            bottom <= bounds[6],
            bottom >= bounds[7],
        ]
    ).T
    eqn_idx, bounds_idx = np.where(mask_arr)
    points = {i: [] for i in range(mask_arr.shape[0])}
    fulcrum = len(eqn_idx) / 2
    for i in range(len(eqn_idx)):
        for b_idx in bounds_idx[eqn_idx == i]:
            if b_idx < fulcrum:
                points[i].append(
                    (
                        int(
                            (eqns[i, 1] * bounds[b_idx] + eqns[i, 2])
                            / (-1 * eqns[i, 0])
                        ),
                        bounds[b_idx],
                    )
                )
            else:
                points[i].append(
                    (bounds[b_idx], int(eqns[i, 0] * bounds[b_idx] + eqns[i, 2]))
                )
    return points


def main(eqns, annotVideoPath=None):
    chkpt_endpts = checkpoint_endpoints(eqns)

    counter = {
        "NBU": 0,
        "NBT": 0,
        "NBR": 0,
        "NBL": 0,
        "SBU": 0, 
        "SBT": 0,
        "SBR": 0, 
        "SBL": 0,
        "EBU": 0, 
        "EBT": 0,
        "EBR": 0, 
        "EBL": 0,
        "WBU": 0, 
        "WBT": 0,
        "WBR": 0, 
        "WBL": 0,
    }

    metric = NearestNeighborDistanceMetric(
        "cosine", matching_threshold=0.2, budget=None
    )
    tracker = Tracker(metric, max_age=30, n_init=2)
    encoder = gdet.create_box_encoder("./mars-small128.pb", batch_size=1)

    VideoCapture = cv2.VideoCapture(VIDEO_PATH)
    model = YOLO(MODEL_FILE_PATH)
    classes = {0: "car", 1: "bicycle", 2: "bus", 3: "lorry"}

    ret, frame = VideoCapture.read()
    height, width, channels = frame.shape

    if annotVideoPath:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        VideoWriter = cv2.VideoWriter(ANNOT_VIDEO_PATH, fourcc, 24, (width, height))

    ret, frame = VideoCapture.read()
    while True:
        if ret:
            # list of ultralytics.yolo.engine.results.Results
            matrix = model(frame)[0].boxes.data.detach().cpu().numpy()

            mask = matrix[:, 4] >= CONFIDENCE_THRESHOLD
            matrix = matrix[mask]

            matrix[:, 2:4] = matrix[:, 2:4] - matrix[:, :2]
            indices = preprocessing.non_max_suppression(
                matrix[:, :4], max_bbox_overlap=0.8, scores=matrix[:, -1]
            )
            matrix = matrix[indices]
            # featureVector shape: [12, 128]
            featureVector = encoder(frame, matrix[:, :4])

            detections = [
                Detection(m[:4], m[4], f_vector)
                for m, f_vector in zip(matrix, featureVector)
            ]

            tracker.predict()
            tracker.update(detections)

            tracks = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr().astype(int)

                id = track.track_id

                centerpoint = (int(bbox[0]+(bbox[2]-bbox[0])/2), int(bbox[1]+(bbox[3]-bbox[1])/2))
                if ((centerpoint[0] < 1610) and (centerpoint[0] > 326)) and ((centerpoint[1] < 757) and (centerpoint[1] > 55)):
                    dist = pt_to_line_distance(centerpoint, eqns)
                    
                    checkpoint = np.where(dist<5)[0]
                    if len(checkpoint) > 1:
                        print("Warning: Multiple checkpoints detected: Only the first one added.")
                    if len(checkpoint) != 0:
                        if approaches[checkpoint[0]] not in track.checkpoints:
                            track.checkpoints.append(approaches[checkpoint[0]])
                            print(f"Checkpoint: {approaches[checkpoint[0]]} added for vehicle: {id}")

                    if len(track.checkpoints) > 1 and not track.counted:
                        counter[dir_map[track.checkpoints[0]][track.checkpoints[1]]] += 1
                        track.counted = True
                        print(counter)

                tracks.append((bbox, id, centerpoint))

            for bbox, id, centerpoint in tracks:
                cv2.rectangle(
                    img=frame,
                    pt1=bbox[:2],
                    pt2=bbox[2:4],
                    color=(20, 255, 57),
                    thickness=2,
                )
                # add text to image
                cv2.putText(
                    img=frame,
                    text=str(id),
                    org=(bbox[0], bbox[1] - OFFSET),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(20, 255, 57),  # bgr
                    thickness=3,
                )
                cv2.circle(
                    img=frame,
                    center=centerpoint,
                    radius=5,
                    color=(20, 255, 57),
                    thickness=-1
                )

            for pt_idx in range(4):
                cv2.line(
                    img=frame,
                    pt1=chkpt_endpts[pt_idx][0],
                    pt2=chkpt_endpts[pt_idx][1],
                    color=(0, 0, 255),
                    thickness=3,
                )

            cv2.imshow("Frame", frame)
            if annotVideoPath:
                VideoWriter.write(frame)

            # If the "esc" key is pressed
            key = cv2.waitKey(0)
            if key == 27:
                break

        else:
            break

        ret, frame = VideoCapture.read()

    VideoCapture.release()
    if annotVideoPath:
        VideoWriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input = np.array(
        [
            [1060, 733, 718, 40.3],
            [427, 322, 633, 22.7],
            [960, 707, 623, 145],
            [1608, 269, 623, 159],
        ]
    )

    eqns, _ = line_equations(input)
    main(eqns)
