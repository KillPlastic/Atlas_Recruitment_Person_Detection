import os
import time
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sklearn.metrics import confusion_matrix
import pandas as pd

# ------------------------------------------------------
# CONFIGURAÇÃO
# ------------------------------------------------------
IMAGE_DIR = "../Data2/test/images"
LABEL_DIR = "../Data2/test/labels"

MODEL_PATH = "content/runs/detect/train2/weights/best.pt"

CONF_THRES = 0.3   # confidence threshold for predictions
IOU_THRES = 0.2    # IoU threshold for matching & for NMS
EXCLUDE = {"test_0008.jpg"}

# ------------------------------------------------------
# FUNÇÃO: carregar ground truth YOLO txt (CORRIGIDA)
# ------------------------------------------------------
def load_yolo_labels(label_path, img_w, img_h):
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, w, h = map(float, parts[:5])
            # converter YOLO (xc,yc,w,h) normalizado -> xyxy em pixels
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)   # *** usar h aqui (era bug antes) ***
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            # assegurar limites dentro da imagem
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            labels.append([x1, y1, x2, y2])
    return labels

# ------------------------------------------------------
# FUNÇÃO: IoU
# ------------------------------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    boxA_area = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxB_area = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = boxA_area + boxB_area - inter
    if union <= 0:
        return 0.0
    return inter / union

# ------------------------------------------------------
# FUNÇÃO: NMS (greedy)
# ------------------------------------------------------
def nms_boxes(boxes, scores=None, iou_threshold=0.5):
    """
    boxes: list of [x1,y1,x2,y2]
    scores: optional list of scores; if None, uses order
    returns: filtered boxes (indices kept)
    """
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes, dtype=float)
    if scores is None:
        scores = np.arange(len(boxes))[::-1]  # reverse order if no scores

    indices = np.argsort(scores)[::-1]  # descending
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        if len(indices) == 1:
            break
        rest = indices[1:]
        ious = np.array([iou(boxes_np[i], boxes_np[j]) for j in rest])
        indices = rest[ious <= iou_threshold]
    return keep

# ------------------------------------------------------
# CARREGAR MODELOS
# ------------------------------------------------------
print("Loading YOLO model...")
yolo_model = YOLO(MODEL_PATH)

print("Loading SAHI model...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRES,
    device="cpu"
)

# ------------------------------------------------------
# LISTA DE IMAGENS
# ------------------------------------------------------
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg','.png','.jpeg')) and f not in EXCLUDE
])

print(f"Found {len(image_files)} images (excluded {EXCLUDE})")

# ------------------------------------------------------
# ARRAYS PARA CONFUSION MATRIX GLOBAL
# ------------------------------------------------------
y_true_global_yolo = []
y_pred_global_yolo = []

y_true_global_sahi = []
y_pred_global_sahi = []

results_summary = []

# ------------------------------------------------------
# LOOP IMAGENS
# ------------------------------------------------------
for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.rsplit(".", 1)[0] + ".txt")

    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARN] Could not load {img_name}, skipping")
        continue
    h, w = image.shape[:2]

    gt_boxes = load_yolo_labels(label_path, w, h)

    # ---------------- YOLO inference (filter by conf) ----------------
    t0 = time.time()
    yres = yolo_model(image)[0]
    yolo_time = time.time() - t0

    yolo_boxes = []
    yolo_scores = []
    for box in yres.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        yolo_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        yolo_scores.append(conf)

    # apply NMS to YOLO boxes (usually YOLO already NMS'd, but safe)
    keep_idx = nms_boxes(yolo_boxes, scores=yolo_scores, iou_threshold=IOU_THRES)
    yolo_boxes = [yolo_boxes[i] for i in keep_idx]
    yolo_scores = [yolo_scores[i] for i in keep_idx]

    # ---------------- SAHI inference (filter by conf and NMS) ----------------
    t0 = time.time()
    sres = get_sliced_prediction(
        image,
        sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
    )
    sahi_time = time.time() - t0

    sahi_boxes_all = []
    sahi_scores_all = []
    for obj in sres.object_prediction_list:
        score = float(obj.score.value)
        if score < CONF_THRES:
            continue
        b = list(map(int, obj.bbox.to_xyxy()))
        sahi_boxes_all.append(b)
        sahi_scores_all.append(score)

    # NMS for SAHI
    keep_idx = nms_boxes(sahi_boxes_all, scores=sahi_scores_all, iou_threshold=IOU_THRES)
    sahi_boxes = [sahi_boxes_all[i] for i in keep_idx]
    sahi_scores = [sahi_scores_all[i] for i in keep_idx]

    # ------------------------------------------------------
    # UPDATE GLOBAL CONFUSION ARRAYS (GT loop then unmatched preds counted as FP)
    # ------------------------------------------------------
    def update_confusion(gt_boxes, pred_boxes, y_true_list, y_pred_list):
        matched_preds = set()
        # GT -> TP or FN
        for gt in gt_boxes:
            matched = False
            for i, p in enumerate(pred_boxes):
                if i in matched_preds:
                    continue
                if iou(gt, p) >= IOU_THRES:
                    matched_preds.add(i)
                    matched = True
                    break
            y_true_list.append(1)
            y_pred_list.append(1 if matched else 0)
        # Remaining preds -> FP
        for i in range(len(pred_boxes)):
            if i not in matched_preds:
                y_true_list.append(0)
                y_pred_list.append(1)

    update_confusion(gt_boxes, yolo_boxes, y_true_global_yolo, y_pred_global_yolo)
    update_confusion(gt_boxes, sahi_boxes, y_true_global_sahi, y_pred_global_sahi)

    results_summary.append({
        "image": img_name,
        "yolo_time": yolo_time,
        "sahi_time": sahi_time,
        "n_gt": len(gt_boxes),
        "n_yolo_pred": len(yolo_boxes),
        "n_sahi_pred": len(sahi_boxes)
    })

# ------------------------------------------------------
# CONFUSION MATRICES GLOBAIS
# ------------------------------------------------------
cm_yolo = confusion_matrix(y_true_global_yolo, y_pred_global_yolo, labels=[1,0])
cm_sahi = confusion_matrix(y_true_global_sahi, y_pred_global_sahi, labels=[1,0])

plt.figure(figsize=(5,5))
sns.heatmap(cm_yolo, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Pred: Person", "Pred: No Person"],
            yticklabels=["GT: Person", "GT: No Person"])
plt.title("YOLO - Global Confusion Matrix")
plt.tight_layout()
plt.savefig("confmat_yolo_global.png")
plt.close()

plt.figure(figsize=(5,5))
sns.heatmap(cm_sahi, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: Person", "Pred: No Person"],
            yticklabels=["GT: Person", "GT: No Person"])
plt.title("SAHI - Global Confusion Matrix")
plt.tight_layout()
plt.savefig("confmat_sahi_global.png")
plt.close()

# ------------------------------------------------------
# SAVE CSV + TIME PLOT
# ------------------------------------------------------
df = pd.DataFrame(results_summary)
df.to_csv("comparison_results.csv", index=False)

sns.set(style="whitegrid")
plt.figure(figsize=(10,5))
sns.barplot(data=df[["yolo_time","sahi_time"]])
plt.title("Inference Time: YOLO vs SAHI")
plt.savefig("time_comparison.png")
plt.close()

print("Saved: confmat_yolo_global.png, confmat_sahi_global.png, time_comparison.png, comparison_results.csv")
print("DONE")
