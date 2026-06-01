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
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "class": "normal skin",
            "confidence": 0.0
        }

    boxes = result.boxes

    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    best_index = confidences.argmax()

    class_id = classes[best_index]
    confidence = float(confidences[best_index])

    predicted_class = model.names[class_id]

    return {
        "class": predicted_class,
        "confidence": confidence
    }