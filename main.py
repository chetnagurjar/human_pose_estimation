
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --------------------------------
# 1. LOAD YOLO POSE MODEL
# --------------------------------

model = YOLO("yolo11n-pose.pt")

# --------------------------------
# 2. VIDEO INPUT AND OUTPUT
# --------------------------------

input_path = "input/dance.mp4"
output_path = "output/skeleton_output.mp4"

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise Exception("Error: Cannot open input video!")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise Exception("Error: Cannot create output video!")

# --------------------------------
# 3. COCO 17 KEYPOINT CONNECTIONS
# --------------------------------

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

# --------------------------------
# 4. PROCESS VIDEO FRAMES
# --------------------------------

confidence_threshold = 0.4

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Create a completely black background
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

    # Process detected people
    for result in results:

        if result.keypoints is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            confidences = result.keypoints.conf.cpu().numpy()
        else:
            confidences = np.ones(
                (len(keypoints), 17)
            )

        # Process each detected person
        for person_index, person in enumerate(keypoints):

            person_conf = confidences[person_index]

            # Draw skeleton connections
            for point1, point2 in skeleton:

                x1, y1 = person[point1]
                x2, y2 = person[point2]

                if (
                    person_conf[point1] > confidence_threshold
                    and person_conf[point2] > confidence_threshold
                ):

                    cv2.line(
                        black_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA
                    )

            # Draw keypoint circles
            for point_index, (x, y) in enumerate(person):

                if person_conf[point_index] > confidence_threshold:

                    cv2.circle(
                        black_frame,
                        (int(x), int(y)),
                        5,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA
                    )

    # Save processed frame
    out.write(black_frame)

    # Display output while processing
    cv2.imshow("Human Pose Estimation", black_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------
# 5. RELEASE RESOURCES
# --------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print("Pose estimation completed!")
print("Output saved at:", output_path)
