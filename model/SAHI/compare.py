# compare_pairs.py

import os
import cv2
import numpy as np

# ------------------------------------------------
# 1. Settings
# ------------------------------------------------
no_sahi_dir = "SAHI/no_SAHI_tests"   # YOLO-only outputs
sahi_dir = "SAHI/SAHI_tests"         # SAHI outputs

# -----------------------------
# Helper: check valid extensions
# -----------------------------
def is_image_file(filename):
    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return any(filename.lower().endswith(ext) for ext in valid_ext)


# ------------------------------------------------
# 2. Get matching filenames
# ------------------------------------------------
no_sahi_files = sorted([f for f in os.listdir(no_sahi_dir) if is_image_file(f)])
sahi_files = sorted([f for f in os.listdir(sahi_dir) if is_image_file(f)])

# Use intersection of file names
matched_files = [f for f in no_sahi_files if f in sahi_files]

if not matched_files:
    raise RuntimeError("No matching image filenames between folders!")

print(f"Found {len(matched_files)} matching images.")

# ------------------------------------------------
# 3. Display images side-by-side
# ------------------------------------------------
for filename in matched_files:

    # Load both images
    img_no_sahi = cv2.imread(os.path.join(no_sahi_dir, filename))
    img_sahi = cv2.imread(os.path.join(sahi_dir, filename))

    if img_no_sahi is None or img_sahi is None:
        print(f"[WARNING] Could not load {filename}, skipping.")
        continue

    # Resize to same height for cleaner comparison
    h1, w1 = img_no_sahi.shape[:2]
    h2, w2 = img_sahi.shape[:2]
    target_h = min(h1, h2)

    img_no_sahi = cv2.resize(img_no_sahi, (int(w1 * target_h / h1), target_h))
    img_sahi = cv2.resize(img_sahi, (int(w2 * target_h / h2), target_h))

    # Concatenate side by side
    combined = np.hstack((img_no_sahi, img_sahi))

    # Show window
    cv2.imshow(f"Comparison: {filename}", combined)

    print(f"Showing {filename} — press any key for next (q to quit).")
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
print("Done.")
