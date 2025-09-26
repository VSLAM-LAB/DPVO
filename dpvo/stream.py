import os
import cv2
import yaml
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain
import pandas as pd

def load_calibration(calibration_yaml: Path):
    fs = cv2.FileStorage(str(calibration_yaml), cv2.FILE_STORAGE_READ)
    def read_real(key: str) -> float:
        node = fs.getNode(key)
        return float(node.real()) if not node.empty() else 0.0

    fx, fy, cx, cy = map(read_real, ["Camera0.fx", "Camera0.fy", "Camera0.cx", "Camera0.cy"])
    k1, k2, p1, p2, k3 = map(read_real, ["Camera0.k1", "Camera0.k2", "Camera0.p1", "Camera0.p2", "Camera0.k3"])
    fs.release()

    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,   1]], dtype=np.float32)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    has_dist = not np.allclose(dist, 0.0)
    return K, dist, has_dist

def image_stream(queue, sequence_path, rgb_csv, calibration_yaml):

    """ image generator """
    K, _, _ = load_calibration(calibration_yaml)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Load paths
    df = pd.read_csv(rgb_csv)       
    image_list = df['path_rgb0'].to_list()
    timestamps = df['ts_rgb0 (s)'].to_list()

    # Load images
    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(os.path.join(sequence_path, imfile)))
       
        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((timestamps[t], image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

