# Yoga Posture Classification

## Project Title

Yoga Posture Classification

---

## Abstract

Yoga Pose Classification is a Machine Learning-based application that identifies yoga poses from uploaded images. The system uses MediaPipe Pose Estimation to extract human body landmarks and calculate joint angles. These angle features are then used by a Random Forest Machine Learning model to classify yoga poses.

The application provides pose prediction, confidence score, pose description, pose benefits, and recommended next poses through a user-friendly web interface developed using Flask.

---

## Problem Statement

Yoga practitioners often struggle to identify whether they are performing the correct yoga pose. This project aims to automatically recognize yoga poses from images using Machine Learning techniques and provide useful information about the detected pose.

---

## Objectives

* Detect yoga poses from uploaded images.
* Extract body landmarks using MediaPipe.
* Calculate joint angles as features.
* Train a Machine Learning model for pose classification.
* Display pose information and recommendations.
* Develop a user-friendly web application.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Flask
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Scikit-Learn
* Joblib

### Machine Learning Algorithm

* Random Forest Classifier

---

## Dataset Description

The dataset contains images of five yoga poses.

### Yoga Pose Classes

1. Tree Pose
2. Warrior II Pose
3. Goddess Pose
4. Plank Pose
5. Downward Dog Pose

### Dataset Structure

dataset/

├── TRAIN/

│ ├── tree/

│ ├── warrior2/

│ ├── goddess/

│ ├── plank/

│ └── downdog/

│

└── TEST/

├── tree/

├── warrior2/

├── goddess/

├── plank/

└── downdog/

---

## Methodology

### Step 1: Dataset Collection

Yoga pose images are collected and organized into separate folders based on pose categories.

### Step 2: Pose Landmark Detection

MediaPipe Pose is used to detect 33 human body landmarks from each image.

### Step 3: Feature Extraction

Joint angles are calculated from important body parts:

* Left Elbow
* Right Elbow
* Left Knee
* Right Knee

These angles become the feature set used for Machine Learning.

### Step 4: Dataset Generation

Extracted features are stored in a CSV file:

csv_files/yoga_dataset.csv

Total Extracted Samples:

1042

### Step 5: Model Training

A Random Forest Classifier is trained using the generated feature dataset.

Training Process:

80% Training Data

20% Testing Data

### Step 6: Model Evaluation

The trained model is evaluated using the testing dataset.

Model Accuracy:

84.21%

### Step 7: Prediction

When a user uploads an image:

1. MediaPipe extracts body landmarks.
2. Joint angles are calculated.
3. Features are passed to the trained model.
4. The yoga pose is predicted.
5. Confidence score is displayed.

---

## System Architecture

Uploaded Image

↓

MediaPipe Pose Detection

↓

33 Body Landmarks

↓

Joint Angle Calculation

↓

Feature Extraction

↓

Random Forest Classifier

↓

Yoga Pose Prediction

↓

Display Results

---

## Features

### Core Features

* Yoga Pose Detection
* Image Upload
* Pose Classification
* Confidence Score Display
* Pose Description

### Additional Features

* Pose Benefits
* Recommended Next Poses
* User-Friendly Interface
* Uploaded Image Preview

---

## Project Structure

YogaPose/

├── csv_files/

│ └── yoga_dataset.csv

├── dataset/

│

├── models/

│ └── yoga_pose_model.pkl

│

├── templates/

│ ├── index.html

│ └── result.html

│

├── uploads/

│

├── app.py

├── extract_dataset.py

├── train_model.py

---

## Installation

Install Required Packages:

pip install flask

pip install opencv-python

pip install mediapipe

pip install numpy

pip install pandas

pip install scikit-learn

pip install joblib

---

## Running the Project

### Step 1

Generate Feature Dataset

python extract_dataset.py

### Step 2

Train the Machine Learning Model

python train_model.py

### Step 3

Run Flask Application

python app.py

### Step 4

Open Browser

http://127.0.0.1:5000

---

## Sample Output

<img width="1916" height="873" alt="Screenshot 2026-05-29 214108" src="https://github.com/user-attachments/assets/b22465ca-856b-416a-a5b0-ce9accfec5ea" />


<img width="1895" height="862" alt="Screenshot 2026-05-29 214140" src="https://github.com/user-attachments/assets/0fd9d24c-7761-4d6d-a0d9-a09e27e8186a" />

Detected Pose:
Tree

Confidence Score:

64.5%

Description:

Tree Pose improves balance, focus and body stability.

Benefits:

* Improves balance
* Strengthens legs
* Improves concentration

Recommended Next Poses:

* Warrior II
* Goddess
* Downward Dog

---

## Advantages

* Lightweight Machine Learning Model
* Fast Prediction
* Easy to Use
* Accurate Pose Recognition
* Real-Time Feature Extraction
* Simple Deployment

---

## Applications

* Fitness Monitoring
* Yoga Training Assistance
* Healthcare and Wellness
* Personal Fitness Tracking
* Educational Projects

---

## Future Enhancements

* Real-Time Webcam Detection
* Pose Correction System
* Personalized Yoga Trainer
* Mobile Application Support
* Additional Yoga Pose Categories
* Voice-Based Guidance

---

## Conclusion

The Yoga Pose Detection System successfully recognizes yoga poses using Machine Learning techniques. MediaPipe is used to extract body landmarks, and a Random Forest Classifier is trained on joint-angle features to classify yoga poses accurately. The system achieved an accuracy of 84.21% and provides useful information such as confidence score, pose description, benefits, and recommendations through a web-based interface.
