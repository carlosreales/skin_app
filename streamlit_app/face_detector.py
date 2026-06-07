import cv2
import numpy as np
from PIL import Image, ImageDraw


def detect_face(image):
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    return len(faces) > 0


def analyze_face_position(image):

    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return {
            "face_detected": False,
            "message": "No se detectó un rostro. Intenta mirar de frente a la cámara y mejorar la iluminación.",
            "quality": "Mala",
            "annotated_image": image
        }

    image_width, image_height = image.size

    faces = sorted(
        faces,
        key=lambda face: face[2] * face[3],
        reverse=True
    )

    x, y, w, h = faces[0]

    face_center_x = x + w / 2
    face_center_y = y + h / 2

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    offset_x = abs(face_center_x - image_center_x) / image_width
    offset_y = abs(face_center_y - image_center_y) / image_height

    face_area_ratio = (w * h) / (image_width * image_height)

    issues = []

    if offset_x > 0.18:
        issues.append("El rostro está muy desplazado horizontalmente.")

    if offset_y > 0.22:
        issues.append("El rostro está muy desplazado verticalmente.")

    if face_area_ratio < 0.08:
        issues.append("El rostro está muy lejos de la cámara.")

    if face_area_ratio > 0.45:
        issues.append("El rostro está demasiado cerca de la cámara.")

    annotated_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated_image)

    draw.rectangle(
        [(x, y), (x + w, y + h)],
        outline="green",
        width=4
    )

    draw.line(
        [(image_center_x - 20, image_center_y), (image_center_x + 20, image_center_y)],
        fill="blue",
        width=3
    )

    draw.line(
        [(image_center_x, image_center_y - 20), (image_center_x, image_center_y + 20)],
        fill="blue",
        width=3
    )

    if len(issues) == 0:
        quality = "Buena"
        message = "Rostro detectado correctamente. La imagen es adecuada para el análisis."
    elif len(issues) <= 2:
        quality = "Aceptable"
        message = "Rostro detectado, pero la captura podría mejorar: " + " ".join(issues)
    else:
        quality = "Mala"
        message = "La captura no es ideal: " + " ".join(issues)

    return {
        "face_detected": True,
        "message": message,
        "quality": quality,
        "annotated_image": annotated_image
    }

def extract_face_region(image, padding=40):
    image_np = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return image

    faces = sorted(
        faces,
        key=lambda face: face[2] * face[3],
        reverse=True
    )

    x, y, w, h = faces[0]

    img_width, img_height = image.size

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(img_width, x + w + padding)
    y2 = min(img_height, y + h + padding)

    face_crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )

    return face_crop