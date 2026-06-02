from PIL import Image, ImageDraw


def draw_detections(image, detections):
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["box"]

        class_name = det.get("class_name", det.get("class", "Sin clase"))
        confidence = det["confidence"] * 100

        label = f"{class_name} {confidence:.1f}%"

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(y1 - 18, 0)),
            label,
            fill="red"
        )

    return image