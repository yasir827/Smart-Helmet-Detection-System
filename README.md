# 🪖 AI-Powered Helmet Detection System using YOLOv8

## 📌 Overview

This project presents an **AI-Powered Helmet Detection System** developed using YOLOv8 and Computer Vision techniques. The system detects motorcycle riders and determines helmet compliance from images, supporting intelligent traffic monitoring and road safety applications.

The project covers the complete deep learning workflow, including dataset acquisition, preprocessing, model training, validation, inference, and performance evaluation.

---

## 🎯 Objectives

* Detect motorcycle riders and helmets in images.
* Improve road safety through automated monitoring.
* Build a robust object detection model using YOLOv8.
* Develop a deployment-ready traffic surveillance solution.

---

## 🏗️ Project Pipeline

```text
Helmet Detection System (YOLOv8)

        │
        ▼
Dataset Collection
        │
        ▼
Dataset Download
(KaggleHub)
        │
        ▼
Dataset Exploration
& Verification
        │
        ▼
Label Inspection
(YOLO Format)
        │
        ▼
Data Configuration
(YAML Setup)
        │
        ▼
Model Selection
(YOLOv8n)
        │
        ▼
Model Training
(50 Epochs)
        │
        ▼
Model Validation
(Precision, Recall, mAP)
        │
        ▼
Best Model Selection
(best.pt)
        │
        ▼
Helmet Detection
on New Images
        │
        ▼
Performance Evaluation
        │
        ▼
Deployment Ready Model
```

---

# 📂 Dataset

## Dataset Name

**Smart Helmet Detection Using Deep Learning Dataset**

## Dataset Source

Dataset downloaded from Kaggle using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "anushkagovindkadam/smart-helmet-detection-using-dl"
)
```

## Dataset Description

The dataset contains annotated traffic and motorcycle rider images with helmet-related labels in YOLO object detection format.

Each image includes bounding box annotations for riders, passengers, motorcycles, and helmet usage conditions.

---

## Dataset Structure

```text
data/
│
├── train/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

---

## Classes

| Class ID | Class Name               |
| -------- | ------------------------ |
| 0        | driver_with_helmet       |
| 1        | bike                     |
| 2        | driver                   |
| 3        | passenger_with_helmet    |
| 4        | passenger                |
| 5        | driver_without_helmet    |
| 6        | passenger_without_helmet |

**Total Classes:** 7

---

## Annotation Format

The dataset follows YOLO annotation format:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.521 0.442 0.184 0.329
```

---

## Data Preparation

The following preprocessing steps were performed:

* Downloaded dataset using KaggleHub
* Verified image and label files
* Inspected class distributions
* Validated annotation integrity
* Configured YOLO YAML file
* Prepared train and validation paths

---

# 🛠️ Technologies Used

* Python
* YOLOv8
* Ultralytics
* OpenCV
* KaggleHub
* Matplotlib
* Deep Learning
* Computer Vision

---

# ⚙️ Model Configuration

| Parameter  | Value              |
| ---------- | ------------------ |
| Model      | YOLOv8n            |
| Task       | Object Detection   |
| Epochs     | 50                 |
| Image Size | 640 × 640          |
| Batch Size | 16                 |
| Patience   | 10                 |
| Framework  | Ultralytics YOLOv8 |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AI-Powered-Helmet-Detection-System-using-YOLOv8.git

cd AI-Powered-Helmet-Detection-System-using-YOLOv8
```

## Install Dependencies

```bash
pip install ultralytics kagglehub opencv-python matplotlib
```

---

# 🏋️ Model Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="helmet.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10
)
```

---

# 📊 Model Evaluation

After training, the model was evaluated using standard object detection metrics.

### Evaluation Metrics

* Precision
* Recall
* mAP@50
* mAP@50-95

Example:

```python
metrics = model.val()

print(metrics.box.map50)
print(metrics.box.map)
print(metrics.box.mp)
print(metrics.box.mr)
```

---

# 🔍 Inference

## Load Trained Model

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

## Predict on New Images

```python
results = model.predict(
    source="test_images/",
    conf=0.25,
    save=True
)
```

---

# 📁 Repository Structure

```text
Helmet-Detection-System/
│
├── dataset/
│
├── notebooks/
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
│
├── helmet.yaml
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
└── assets/
    ├── pipeline.png
    ├── results.png
    └── predictions.png
```

---

# 🎯 Applications

* Smart Traffic Monitoring
* Road Safety Enforcement
* Intelligent Transportation Systems
* Traffic Violation Detection
* Urban Surveillance
* Smart City Solutions

---

# ✅ Project Results

* Successfully trained YOLOv8 object detection model.
* Detected helmet compliance among riders and passengers.
* Generated bounding box predictions on unseen images.
* Evaluated model using Precision, Recall, and mAP metrics.
* Developed a deployment-ready detection pipeline.

---

# 🔮 Future Improvements

* Real-time CCTV integration
* Video-based helmet detection
* Traffic violation alert system
* Multi-camera monitoring
* Edge AI deployment
* Smart city dashboard integration



# ⭐ Acknowledgements

* Ultralytics YOLOv8
* KaggleHub
* OpenCV Community
* Computer Vision Research Community
* Smart Helmet Detection Dataset Contributors
