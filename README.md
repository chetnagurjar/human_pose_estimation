# Human Pose Estimation Using YOLO and OpenCV

## 📌 Project Overview

This project implements Human Pose Estimation using Computer Vision techniques to detect and track human body movements from video input.

Using **YOLO Pose** and **OpenCV**, the system identifies 17 human body keypoints based on the COCO dataset and visualizes the detected body structure as a skeleton.

The original video frame is replaced with a completely black background, displaying only white keypoints (dots) and connecting lines. This makes it possible to visualize human movement without displaying the person's actual appearance.

## 🎯 Objectives

- Perform human pose estimation on a video containing human movement.
- Detect 17 human body keypoints using the COCO format.
- Identify body joints such as shoulders, elbows, wrists, hips, knees, and ankles.
- Connect detected keypoints to generate a human skeleton.
- Remove the original video background and person.
- Generate a skeleton-only output video with a black background and white keypoints.

## 🛠️ Technologies Used

- **Python** – Main programming language.
- **YOLO Pose (Ultralytics)** – Human pose detection and keypoint extraction.
- **OpenCV** – Video processing, frame manipulation, and skeleton visualization.
- **NumPy** – Numerical operations and black-frame generation.

## 🧠 COCO 17 Keypoints

The project uses the COCO human pose estimation format, which consists of 17 body keypoints:

| Index | Keypoint |
|---|---|
| 0 | Nose |
| 1 | Left Eye |
| 2 | Right Eye |
| 3 | Left Ear |
| 4 | Right Ear |
| 5 | Left Shoulder |
| 6 | Right Shoulder |
| 7 | Left Elbow |
| 8 | Right Elbow |
| 9 | Left Wrist |
| 10 | Right Wrist |
| 11 | Left Hip |
| 12 | Right Hip |
| 13 | Left Knee |
| 14 | Right Knee |
| 15 | Left Ankle |
| 16 | Right Ankle |

## ⚙️ Project Workflow

1. **Video Input:** Load a video containing a person performing movements or dancing.
2. **Pose Detection:** Process each video frame using the pretrained YOLO Pose model.
3. **Keypoint Extraction:** Extract the coordinates and confidence scores of 17 COCO keypoints.
4. **Skeleton Generation:** Connect the detected body joints using lines.
5. **Background Replacement:** Create a black frame and draw only the detected skeleton.
6. **Video Generation:** Combine the processed frames and save them as an output video.

## 📂 Project Structure

```text
Human-Pose-Estimation/
│
├── input/
│   └── dance.mp4
│
├── output/
│   └── skeleton_output.mp4
│
├── main.py
├── requirements.txt
└── README.md
```

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd Human-Pose-Estimation
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Input Video

Place your video inside the `input` folder and name it:

```text
dance.mp4
```

### 5. Run the Program

```bash
python main.py
```

The pretrained YOLO Pose model will be downloaded automatically when required.

## 📊 Output

The program generates a video containing:

- A completely black background.
- White dots representing detected body joints.
- White lines connecting the joints.
- Skeleton movements corresponding to the person's movements in the original video.
- No visible original person or video background.

**Output location:**

```text
output/skeleton_output.mp4
```

## 🔍 Applications

Human Pose Estimation can be used in several areas:

- Dance movement analysis.
- Fitness and exercise tracking.
- Human activity recognition.
- Sports performance analysis.
- Gesture recognition.
- Motion analysis and animation.

## 📚 References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [COCO Dataset](https://cocodataset.org/)

