import os, time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ---------- CONFIG ----------
IMAGE_DIR = "../Data2/test/images"
LABEL_DIR = "../Data2/test/labels"
MODEL_PATH = "content/runs/detect/train2/weights/best.pt"

CONF_THRES = 0.3   # adjust to reproduce visual outputs
IOU_THRES = 0.2
EXCLUDE = {"test_0008.jpg"}

APPLY_NMS = True    # set False to match visual outputs that showed many boxes
NMS_IOU = 0.5

# ---------- HELPERS ----------
def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls, xc, yc, w, h = map(float, parts[:5])
            x1 = int((xc - w/2)*img_w); y1 = int((yc - h/2)*img_h)
            x2 = int((xc + w/2)*img_w); y2 = int((yc + h/2)*img_h)
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(img_w-1,x2), min(img_h-1,y2)
            boxes.append([x1,y1,x2,y2])
    return boxes

def iou(a,b):
    xA = max(a[0],b[0]); yA = max(a[1],b[1])
    xB = min(a[2],b[2]); yB = min(a[3],b[3])
    inter_w = max(0, xB-xA); inter_h = max(0, yB-yA)
    inter = inter_w*inter_h
    if inter==0: return 0.0
    areaA = max(0,(a[2]-a[0]))*max(0,(a[3]-a[1]))
    areaB = max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))
    union = areaA+areaB-inter
    return inter/union if union>0 else 0.0

def nms_indices(boxes, scores=None, iou_thresh=0.5):
    if len(boxes)==0: return []
    arr = np.array(boxes, dtype=float)
    if scores is None:
        scores = np.arange(len(boxes))[::-1]
    idxs = np.argsort(scores)[::-1]
    keep=[]
    while len(idxs)>0:
        i = idxs[0]; keep.append(i)
        if len(idxs)==1: break
        rest = idxs[1:]
        ious = np.array([iou(arr[i], arr[j]) for j in rest])
        idxs = rest[ious<=iou_thresh]
    return keep

# ---------- LOAD MODELS ----------
print("Loading models...")
yolo = YOLO(MODEL_PATH)
sahi = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=MODEL_PATH,
                                          confidence_threshold=CONF_THRES, device="cpu")

# ---------- LOOP ----------
image_files = sorted([f for f in os.listdir(IMAGE_DIR)
                      if f.lower().endswith(('.jpg','.png','.jpeg')) and f not in EXCLUDE])
rows=[]

global_y_true_yolo=[]; global_y_pred_yolo=[]
global_y_true_sahi=[]; global_y_pred_sahi=[]

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    lab_path = os.path.join(LABEL_DIR, img_name.rsplit(".",1)[0]+".txt")
    img = cv2.imread(img_path)
    if img is None:
        print("Could not load", img_name); continue
    h,w = img.shape[:2]
    gt = load_yolo_labels(lab_path, w, h)

    # YOLO
    t0=time.time()
    yres = yolo(img)[0]
    t_y = time.time()-t0
    y_boxes=[]; y_scores=[]
    for box in yres.boxes:
        conf = float(box.conf[0])
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        if conf < CONF_THRES: continue
        y_boxes.append([int(x1),int(y1),int(x2),int(y2)])
        y_scores.append(conf)
    if APPLY_NMS:
        keep = nms_indices(y_boxes, y_scores, NMS_IOU)
        y_boxes = [y_boxes[i] for i in keep]; y_scores=[y_scores[i] for i in keep]

    # SAHI
    t0=time.time()
    sres = get_sliced_prediction(img, sahi, slice_height=640, slice_width=640,
                                overlap_height_ratio=0.25, overlap_width_ratio=0.25)
    t_s = time.time()-t0
    s_boxes=[]; s_scores=[]
    for obj in sres.object_prediction_list:
        conf = float(obj.score.value)
        b = list(map(int, obj.bbox.to_xyxy()))
        if conf < CONF_THRES: continue
        s_boxes.append(b); s_scores.append(conf)
    if APPLY_NMS:
        keep = nms_indices(s_boxes, s_scores, NMS_IOU)
        s_boxes = [s_boxes[i] for i in keep]; s_scores=[s_scores[i] for i in keep]

    # Build per-image TP/FP/FN using matching IoU >= IOU_THRES
    def per_image_confusion(gt_boxes, pred_boxes):
        y_true=[]; y_pred=[]
        matched=set()
        for gt in gt_boxes:
            matched_flag=False
            for i,p in enumerate(pred_boxes):
                if i in matched: continue
                if iou(gt,p) >= IOU_THRES:
                    matched.add(i); matched_flag=True; break
            y_true.append(1); y_pred.append(1 if matched_flag else 0)
        for i in range(len(pred_boxes)):
            if i not in matched:
                y_true.append(0); y_pred.append(1)
        return y_true, y_pred

    yt_y, yp_y = per_image_confusion(gt, y_boxes)
    yt_s, yp_s = per_image_confusion(gt, s_boxes)

    global_y_true_yolo.extend(yt_y); global_y_pred_yolo.extend(yp_y)
    global_y_true_sahi.extend(yt_s); global_y_pred_sahi.extend(yp_s)

    TP_y = sum(1 for a,b in zip(yt_y,yp_y) if a==1 and b==1)
    FN_y = sum(1 for a,b in zip(yt_y,yp_y) if a==1 and b==0)
    FP_y = sum(1 for a,b in zip(yt_y,yp_y) if a==0 and b==1)

    TP_s = sum(1 for a,b in zip(yt_s,yp_s) if a==1 and b==1)
    FN_s = sum(1 for a,b in zip(yt_s,yp_s) if a==1 and b==0)
    FP_s = sum(1 for a,b in zip(yt_s,yp_s) if a==0 and b==1)

    rows.append({"image":img_name, "n_gt":len(gt), "n_yolo_pred":len(y_boxes), "n_sahi_pred":len(s_boxes),
                 "TP_y":TP_y, "FP_y":FP_y, "FN_y":FN_y, "TP_s":TP_s, "FP_s":FP_s, "FN_s":FN_s,
                 "y_time":t_y, "s_time":t_s})

# ---------- AGGREGATE ----------
df = pd.DataFrame(rows)
df.to_csv("detailed_TP_FP_FN_per_image.csv", index=False)

# global totals
totals = {
    "YOLO_TP": df["TP_y"].sum(), "YOLO_FP": df["FP_y"].sum(), "YOLO_FN": df["FN_y"].sum(),
    "SAHI_TP": df["TP_s"].sum(), "SAHI_FP": df["FP_s"].sum(), "SAHI_FN": df["FN_s"].sum()
}
print("Totals:", totals)
print("Wrote detailed_TP_FP_FN_per_image.csv")
