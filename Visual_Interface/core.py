from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np

MODEL_CACHE = {}

def plain_predict_image(image: np.ndarray, model) -> np.ndarray:
    results = model(image, verbose=False)[0]

    annotated = image.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return annotated


def sahi_predict_image(
    image: np.ndarray,
    detection_model,
    slice_size=640,
    overlap=0.2,
) -> np.ndarray:

    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )

    annotated = image.copy()

    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        label = f"{obj.category.name} {obj.score.value:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return annotated


def load_model(model_path: str, use_sahi: bool):
    key = (model_path, use_sahi)

    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if use_sahi:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=0.16,
            device="cpu",
        )
    else:
        model = YOLO(model_path)

    MODEL_CACHE[key] = model
    return model

def run_inference(
    image: np.ndarray,
    model_path: str,
    use_sahi: bool,
    slice_size: int = 640,
    overlap: float = 0.2,
):
    model = load_model(model_path, use_sahi)

    if use_sahi:
        return sahi_predict_image(image, model, slice_size, overlap)
    else:
        return plain_predict_image(image, model)


def webcam_stream(
    frame,
    model_path,
    use_sahi,
    slice_size,
    overlap,
):
    return run_inference(
        frame,
        model_path,
        use_sahi,
        slice_size,
        overlap,
    )
