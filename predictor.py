from ultralytics import YOLO
import numpy as np
import streamlit as st


MODEL_PATH = "models/dermavision_best_v3.pt"


@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model


def predict(image):
    model = load_model()

    image_array = np.array(image.convert("RGB"))

    results = model.predict(
        source=image_array,
        conf=0.15,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": confidence
            })

    if len(detections) == 0:
        return {
            "class": "Sin detección",
            "confidence": 0.0,
            "detections": []
        }

    detections = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True
    )

    best_detection = detections[0]

    return {
        "class": best_detection["class"],
        "confidence": best_detection["confidence"],
        "detections": detections
    }