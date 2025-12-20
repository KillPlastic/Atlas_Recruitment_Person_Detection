# sahi_batch_inference.py

import cv2
import os
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "../Data2/test/images"
OUTPUT_DIR = "SAHI/SAHI_tests"
MODEL_PATH = "content/runs/detect/train2/weights/best.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load SAHI-wrapped YOLO11
# -----------------------------
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=0.16,
    device="cpu",       # or "cuda:0"
)

# -----------------------------
# VALID IMAGE EXTENSIONS
# -----------------------------
valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------
# PROCESS EVERY IMAGE
# -----------------------------
for filename in os.listdir(IMAGE_DIR):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue

    input_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(input_path)

    if image is None:
        print(f"Could not load {filename}")
        continue

    print(f"Processing: {filename}")

    # -----------------------------
    # 2. SAHI sliced inference
    # -----------------------------
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # -----------------------------
    # 3. Draw results manually
    # -----------------------------
    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        label = f"{obj.category.name} {obj.score.value:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # -----------------------------
    # 4. Save output
    # -----------------------------
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

print("Batch SAHI inference complete.")
