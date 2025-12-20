import gradio as gr
import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CONFIG_FILE = "last_model.json"
DEFAULT_CONF = 0.16

# -------------------------------------------------
# MODEL CACHE
# -------------------------------------------------
model_cache = {
    "path": None,
    "yolo": None,
    "sahi": None
}

# -------------------------------------------------
# LOAD / SAVE LAST MODEL PATH
# -------------------------------------------------
def load_last_model_path():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f).get("model_path")
    return None

def save_last_model_path(path):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"model_path": path}, f)

# -------------------------------------------------
# MODEL LOADER
# -------------------------------------------------
def load_model(model_path):
    if model_cache["path"] == model_path:
        return

    print(f"Loading model: {model_path}")

    model_cache["yolo"] = YOLO(model_path)

    model_cache["sahi"] = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=DEFAULT_CONF,
        device="cpu"
    )

    model_cache["path"] = model_path
    save_last_model_path(model_path)

# -------------------------------------------------
# YOLO INFERENCE
# -------------------------------------------------
def yolo_predict(image):
    results = model_cache["yolo"](image, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{results.names[cls]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return image

# -------------------------------------------------
# SAHI INFERENCE
# -------------------------------------------------
def sahi_predict(image, slice_size, overlap):
    result = get_sliced_prediction(
        image=image,
        detection_model=model_cache["sahi"],
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )

    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        label = f"{obj.category.name} {obj.score.value:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return image

# -------------------------------------------------
# IMAGE PIPELINE
# -------------------------------------------------
def run_image(image, model_file, use_sahi, slice_size, overlap):
    if image is None:
        raise gr.Error("Por favor seleciona uma imagem")

    if model_file is None:
        raise gr.Error("Por favor seleciona um modelo (.pt)")

    load_model(model_file.name)

    image = image.copy()

    if use_sahi:
        return sahi_predict(image, slice_size, overlap)
    else:
        return yolo_predict(image)

# -------------------------------------------------
# UI
# -------------------------------------------------
last_model = load_last_model_path()

with gr.Blocks(title="YOLO / SAHI Detection") as demo:

    gr.Markdown("## Interface de Deteção (YOLO / SAHI)")

    with gr.Row():
        model_file = gr.File(
            label="Selecionar modelo (.pt)",
            file_types=[".pt"],
            value=last_model
        )
        use_sahi = gr.Checkbox(label="Usar SAHI", value=True)

    with gr.Row(visible=True) as sahi_controls:
        slice_size = gr.Slider(
            256, 1024, value=640, step=64,
            label="Slice size"
        )
        overlap = gr.Slider(
            0.0, 0.5, value=0.2, step=0.05,
            label="Overlap"
        )

    img_in = gr.Image(type="numpy", label="Imagem de entrada")
    img_out = gr.Image(label="Resultado")

    btn = gr.Button("Run Detection")

    btn.click(
        run_image,
        inputs=[img_in, model_file, use_sahi, slice_size, overlap],
        outputs=img_out
    )

    # Mostrar / esconder parâmetros SAHI
    def toggle_sahi_controls(use_sahi):
        return gr.update(visible=use_sahi)

    use_sahi.change(
        toggle_sahi_controls,
        inputs=use_sahi,
        outputs=sahi_controls
    )

demo.launch(share=True)

