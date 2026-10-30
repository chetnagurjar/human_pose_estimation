
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ==========================================
# 1. LOAD YOLO POSE MODEL
# ==========================================

print("Loading YOLO Pose Model...")

model = YOLO("yolo11n-pose.pt")

print("Model loaded successfully!")

# ==========================================
# 2. VIDEO INPUT AND OUTPUT
# ==========================================

input_path = "input/dance.mp4"
output_path = "output/skeleton_output.mp4"

os.makedirs("output", exist_ok=True)
os.makedirs("results", exist_ok=True)

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise Exception("Cannot open input video!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

print("Video loaded!")
print("Resolution:", width, "x", height)
print("FPS:", fps)

# ==========================================
# 3. VIDEO WRITER
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise Exception("Cannot create output video!")

# ==========================================
# 4. COCO 17 KEYPOINT CONNECTIONS
# ==========================================

skeleton = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

CONFIDENCE_THRESHOLD = 0.4

# ==========================================
# 5. PROCESS VIDEO FRAME BY FRAME
# ==========================================

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Create completely black background
    black_frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # Perform pose estimation
    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    # ======================================
    # 6. EXTRACT KEYPOINTS
    # ======================================

    for result in results:

        if result.keypoints is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf

        if confidences is not None:
            confidences = confidences.cpu().numpy()
        else:
            confidences = np.ones(
                (len(keypoints), 17)
            )

        # Process every detected person
        for person_index, person in enumerate(keypoints):

            person_conf = confidences[person_index]

            # ==================================
            # 7. DRAW SKELETON CONNECTIONS
            # ==================================

            for point1, point2 in skeleton:

                if (
                    person_conf[point1] > CONFIDENCE_THRESHOLD
                    and person_conf[point2] > CONFIDENCE_THRESHOLD
                ):

                    x1, y1 = person[point1]
                    x2, y2 = person[point2]

                    cv2.line(
                        black_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA
                    )

            # ==================================
            # 8. DRAW WHITE KEYPOINT DOTS
            # ==================================

            for point_index, (x, y) in enumerate(person):

                if person_conf[point_index] > CONFIDENCE_THRESHOLD:

                    cv2.circle(
                        black_frame,
                        (int(x), int(y)),
                        5,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA
                    )

    # ======================================
    # 9. SAVE PROCESSED FRAME
    # ======================================

    out.write(black_frame)

    # Display processed video
    cv2.imshow(
        "Human Pose Estimation",
        black_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_count % 30 == 0:
        print("Processed frames:", frame_count)

# ==========================================
# 10. RELEASE RESOURCES
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nPose Estimation Completed!")
print("Total frames:", frame_count)
print("Output saved:", output_path)
