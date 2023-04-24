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
    Return the equations of a line in the form ax + by + c = 0.

    Parameters
    ----------
    input : ndarray
        A 4x4 matrix with each entry referring to a line in the
        form [x, y, line-length, angle], where the angle is measured
        in degrees counter-clockwise from the horizontal.

    Returns
    -------
    ndarray
        A 4x3 matrix of 4 equations and 3 coefficients in the form ax+by+c=0

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
    return eqns


def pt_to_line_distance(pt, eqns):
    return np.abs(eqns[:, 0] * pt[0] + eqns[:, 1] * pt[1] + eqns[:, 2]) / np.sqrt(
        np.power(eqns[:, 0], 2) + np.power(eqns[:, 1], 2)
    )


def checkpoint_endpoints(eqns, res=(1920, 1080)):
    """
    Return a pair of points for each equation of the line corresponding
    to the coordinates at which the line touches the boundary of image.

    Parameters
    ----------
    eqns : ndarray
        A 4x3 matrix of 4 equations and 3 coefficients in the form ax+by+c=0
    res : tuple
        The resolution of the image in the form (width, height)

    Returns
    -------
    Dict[int: List[Tuple]]
        Integer keys corresponding to each of the 4 equations linked with
        a list of coordinates representing the points of intersection
        between the line equations and the edges of the image.

    """
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


def detection_edges(eqns, res=(1920, 1080)):
    """
    Return the coordinates in the form: [xmin, xmax, ymin, ymax]
    corresponding to the horizontal and vertical axes that if the
    centerpoint an object (vehicle) falls below or above, will not
    count towards the traffic counter.

    Parameters
    ----------
    eqns : ndarray
        A 4x3 matrix of 4 equations and 3 coefficients in the form ax+by+c=0
    res : tuple
        The resolution of the image in the form (width, height)

    Returns
    -------
    ndarray
        Boundary axes in the form: [xmin, xmax, ymin, ymax]

    """
    xs = np.zeros((4, 4))
    ys = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i != j:
                xs[i, j] = (eqns[j, 2] - eqns[i, 2]) / (eqns[i, 0] - eqns[j, 0])
    for i in range(4):
        for j in range(4):
            if i != j:
                ys[i, j] = eqns[i, 0] * xs[i, j] + eqns[i, 2]
    xs = xs.astype(int)
    ys = ys.astype(int)
    mask = (xs < res[0]) & (xs > 0) & (ys < res[1]) & (ys > 0)
    return np.array([xs[mask].min(), xs[mask].max(), ys[mask].min(), ys[mask].max()])


def visualize(frame, tracks, counter, checkpoint_endpoints):
    legend_dim = [743, 827, 434, 253]
    hi_light_color = (55, 214, 234)
    hi_light_color2 = (242, 246, 52)
    main_color = (195, 129, 32)

    for bbox, id, centerpoint in tracks:
        cv2.rectangle(
            img=frame,
            pt1=bbox[:2],
            pt2=bbox[2:4],
            color=hi_light_color,
            thickness=2,
        )
        # add text to image
        cv2.putText(
            img=frame,
            text=str(id),
            org=(bbox[0], bbox[1] - OFFSET),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=hi_light_color,  # bgr
            thickness=3,
        )
        cv2.circle(
            img=frame,
            center=centerpoint,
            radius=5,
            color=hi_light_color,
            thickness=-1,
        )

    for pt_idx in range(len(checkpoint_endpoints)):
        cv2.line(
            img=frame,
            pt1=checkpoint_endpoints[pt_idx][0],
            pt2=checkpoint_endpoints[pt_idx][1],
            color=hi_light_color2,
            thickness=3,
        )

    cv2.rectangle(
        img=frame,
        pt1=legend_dim[:2],
        pt2=(legend_dim[0] + legend_dim[2], legend_dim[1] + legend_dim[3]),
        color=(246, 249, 250),
        thickness=-1,
    )
    cv2.putText(
        img=frame,
        text="NIXIE: Traffic Counter",
        org=(legend_dim[0] + OFFSET, legend_dim[1] + 30),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1,
        color=main_color,
        thickness=3,
    )
    for i, direct in enumerate(["NB", "SB", "EB", "WB"]):
        cv2.putText(
            img=frame,
            text=f"{direct}T: {counter[(direct+'T')]} | {direct}R: {counter[direct+'R']} | {direct}L: {counter[direct+'L']}",
            org=(legend_dim[0] + OFFSET, legend_dim[1] + 80 + 50 * i),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=main_color,
            thickness=3,
        )


def main(eqns, annotVideoPath=None):
    chkpt_endpts = checkpoint_endpoints(eqns)
    edges = detection_edges(eqns)

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

                centerpoint = (
                    int(bbox[0] + (bbox[2] - bbox[0]) / 2),
                    int(bbox[1] + (bbox[3] - bbox[1]) / 2),
                )
                if ((centerpoint[0] < edges[1]) and (centerpoint[0] > edges[0])) and (
                    (centerpoint[1] < edges[3]) and (centerpoint[1] > edges[2])
                ):
                    dist = pt_to_line_distance(centerpoint, eqns)

                    checkpoint = np.where(dist < 5)[0]
                    if len(checkpoint) > 1:
                        print(
                            "Warning: Multiple checkpoints detected: Only the first one added."
                        )
                    if len(checkpoint) != 0:
                        if approaches[checkpoint[0]] not in track.checkpoints:
                            track.checkpoints.append(approaches[checkpoint[0]])

                    if len(track.checkpoints) > 1 and not track.counted:
                        counter[
                            dir_map[track.checkpoints[0]][track.checkpoints[1]]
                        ] += 1
                        track.counted = True

                tracks.append((bbox, id, centerpoint))

            visualize(frame, tracks, counter, chkpt_endpts)

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

    eqns = line_equations(input)
    main(eqns)
