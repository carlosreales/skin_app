from ultralytics import YOLO
import streamlit as st


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "dermavision_best_v3.pt"


@st.cache_resource
def load_model():
    model = YOLO(str(MODEL_PATH))
    return model


def predict(image, conf_threshold=0.15, iou_threshold=0.45):
    model = load_model()

    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = model.names[cls_id]

            detections.append({
                "class_id": cls_id,
                "class": class_name,
                "class_name": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

    if len(detections) == 0:
        return {
            "class": "Sin detección",
            "confidence": 0,
            "detections": []
        }

    best_detection = max(
        detections,
        key=lambda item: item["confidence"]
    )

    return {
        "class": best_detection["class"],
        "confidence": best_detection["confidence"],
        "detections": detections
    }