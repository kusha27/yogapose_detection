import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose


def angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1]-b[1],
        c[0]-b[0]
    ) - np.arctan2(
        a[1]-b[1],
        a[0]-b[0]
    )

    ang = np.abs(radians * 180 / np.pi)

    if ang > 180:
        ang = 360 - ang

    return ang


dataset_path = r"dataset/archive/DATASET/TRAIN"

rows = []

with mp_pose.Pose(static_image_mode=True) as pose:

    for label in os.listdir(dataset_path):

        folder = os.path.join(dataset_path, label)

        if not os.path.isdir(folder):
            continue

        for image_name in os.listdir(folder):

            image_path = os.path.join(folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            result = pose.process(rgb)

            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks.landmark

            try:

                features = [

                    angle(
                        [lm[11].x, lm[11].y],
                        [lm[13].x, lm[13].y],
                        [lm[15].x, lm[15].y]
                    ),

                    angle(
                        [lm[12].x, lm[12].y],
                        [lm[14].x, lm[14].y],
                        [lm[16].x, lm[16].y]
                    ),

                    angle(
                        [lm[23].x, lm[23].y],
                        [lm[25].x, lm[25].y],
                        [lm[27].x, lm[27].y]
                    ),

                    angle(
                        [lm[24].x, lm[24].y],
                        [lm[26].x, lm[26].y],
                        [lm[28].x, lm[28].y]
                    ),

                    label
                ]

                rows.append(features)

            except:
                pass

df = pd.DataFrame(
    rows,
    columns=[
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
        "label"
    ]
)

os.makedirs("csv_files", exist_ok=True)

df.to_csv(
    "csv_files/yoga_dataset.csv",
    index=False
)

print("Dataset Created")
print(df.shape)