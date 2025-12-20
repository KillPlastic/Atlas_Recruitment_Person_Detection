# yolo_image_inference.py

import os
import cv2
from ultralytics import YOLO

# ------------------------------------------------
# 1. Settings
# ------------------------------------------------
input_dir = "../Data2/test/images"              # <-- folder containing images
output_dir = "SAHI/no_SAHI_tests"   # <-- outputs will be saved here
model_path = "content/runs/detect/train2/weights/best.pt"


# Create output folder if missing
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------
# 2. Load YOLO model
# ------------------------------------------------
model = YOLO(model_path)


# ------------------------------------------------
# 3. Helper: check if file is image
# ------------------------------------------------
def is_image_file(filename):
    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in valid_ext)


# ------------------------------------------------
# 4. Process all images in directory
# ------------------------------------------------
image_files = [f for f in os.listdir(input_dir) if is_image_file(f)]

print(f"Found {len(image_files)} images in {input_dir}")

for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)

    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARNING] Could not load {img_name}, skipping.")
        continue

    # -------------------------
    # 5. YOLO Inference
    # -------------------------
    results = model(image)[0]

    # -------------------------
    # 6. Draw predictions
    # -------------------------
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # -------------------------
    # 7. Save output
    # -------------------------
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, image)
    print(f"Saved: {save_path}")

print("\nDone. YOLO predictions saved in:", output_dir)
