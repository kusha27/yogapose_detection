# from flask import Flask, render_template, request
# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import joblib

# app = Flask(__name__)

# model = joblib.load("models/yoga_pose_model.pkl")

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# mp_pose = mp.solutions.pose


# def angle(a, b, c):

#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     radians = np.arctan2(
#         c[1]-b[1],
#         c[0]-b[0]
#     ) - np.arctan2(
#         a[1]-b[1],
#         a[0]-b[0]
#     )

#     ang = np.abs(radians * 180 / np.pi)

#     if ang > 180:
#         ang = 360 - ang

#     return ang


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():

#     file = request.files["image"]

#     filepath = os.path.join(
#         UPLOAD_FOLDER,
#         file.filename
#     )

#     file.save(filepath)

#     image = cv2.imread(filepath)

#     rgb = cv2.cvtColor(
#         image,
#         cv2.COLOR_BGR2RGB
#     )

#     with mp_pose.Pose(
#         static_image_mode=True
#     ) as pose:

#         result = pose.process(rgb)

#         if not result.pose_landmarks:
#             return "No pose detected"

#         lm = result.pose_landmarks.landmark

#         features = [[

#             angle(
#                 [lm[11].x, lm[11].y],
#                 [lm[13].x, lm[13].y],
#                 [lm[15].x, lm[15].y]
#             ),

#             angle(
#                 [lm[12].x, lm[12].y],
#                 [lm[14].x, lm[14].y],
#                 [lm[16].x, lm[16].y]
#             ),

#             angle(
#                 [lm[23].x, lm[23].y],
#                 [lm[25].x, lm[25].y],
#                 [lm[27].x, lm[27].y]
#             ),

#             angle(
#                 [lm[24].x, lm[24].y],
#                 [lm[26].x, lm[26].y],
#                 [lm[28].x, lm[28].y]
#             )

#         ]]

#         prediction = model.predict(features)[0]

#         confidence = (
#             max(model.predict_proba(features)[0])
#             * 100
#         )

#     return render_template(
#         "result.html",
#         prediction=prediction,
#         confidence=round(confidence, 2)
#     )


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

import cv2
import mediapipe as mp
import numpy as np
import os
import joblib

app = Flask(__name__)

# -----------------------------
# Yoga Pose Information
# -----------------------------

POSE_INFO = {

    "tree":
    "Tree Pose improves balance, focus and body stability.",

    "warrior2":
    "Warrior II strengthens legs and improves endurance.",

    "plank":
    "Plank strengthens the core, shoulders and back muscles.",

    "goddess":
    "Goddess Pose improves flexibility and lower body strength.",

    "downdog":
    "Downward Dog stretches the spine, calves and hamstrings."
}


POSE_BENEFITS = {

    "tree": [
        "Improves balance",
        "Strengthens legs",
        "Improves concentration"
    ],

    "warrior2": [
        "Strengthens legs",
        "Improves stamina",
        "Opens hips"
    ],

    "plank": [
        "Strengthens core",
        "Improves posture",
        "Builds endurance"
    ],

    "goddess": [
        "Improves flexibility",
        "Strengthens thighs",
        "Improves stability"
    ],

    "downdog": [
        "Stretches hamstrings",
        "Strengthens shoulders",
        "Improves blood circulation"
    ]
}


RECOMMENDATIONS = {

    "tree": [
        "Warrior II",
        "Goddess",
        "Downward Dog"
    ],

    "warrior2": [
        "Tree",
        "Plank",
        "Downward Dog"
    ],

    "plank": [
        "Downward Dog",
        "Warrior II",
        "Tree"
    ],

    "goddess": [
        "Tree",
        "Warrior II",
        "Plank"
    ],

    "downdog": [
        "Tree",
        "Plank",
        "Warrior II"
    ]
}
# -----------------------------
# Load Model
# -----------------------------

model = joblib.load(
    "models/yoga_pose_model.pkl"
)

# -----------------------------
# Upload Folder
# -----------------------------

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# -----------------------------
# MediaPipe
# -----------------------------

mp_pose = mp.solutions.pose

# -----------------------------
# Angle Function
# -----------------------------

def angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    ang = np.abs(
        radians * 180.0 / np.pi
    )

    if ang > 180:
        ang = 360 - ang

    return ang

# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# -----------------------------
# Predict Pose
# -----------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return "Invalid image"

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    with mp_pose.Pose(
        static_image_mode=True
    ) as pose:

        result = pose.process(rgb)

        if not result.pose_landmarks:
            return """
            <h2>No pose detected.</h2>
            <a href='/'>Try another image</a>
            """

        lm = result.pose_landmarks.landmark

        features = [[

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
            )

        ]]

        prediction = model.predict(
            features
        )[0]

        confidence = max(
            model.predict_proba(
                features
            )[0]
        ) * 100

        description = POSE_INFO.get(
            prediction.lower(),
            "Yoga pose detected."
        )
        
        benefits = POSE_BENEFITS.get(
            prediction.lower(),
            []
        )

        recommendations = RECOMMENDATIONS.get(
            prediction.lower(),
            []
        )
        
        status = "Pose Detected Successfully"


    return render_template(
        "result.html",
        prediction=prediction.upper(),
        confidence=round(
            confidence,
            2
        ),
        description=description,
        benefits=benefits,
        recommendations=recommendations,
        status=status,
        image_name=file.filename
    )

# -----------------------------
# Show Uploaded Image
# -----------------------------

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )

# -----------------------------
# Run App
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )