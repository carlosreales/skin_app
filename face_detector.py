import cv2
import numpy as np


def detect_face(pil_image):

    # Convertir cualquier imagen a RGB
    pil_image = pil_image.convert("RGB")

    image = np.array(pil_image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    return len(faces) > 0