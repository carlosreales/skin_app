import random

CLASSES = [
    "acne",
    "dark circle",
    "darkspot",
    "dry",
    "normal skin",
    "oily",
    "pores",
    "skin redness",
    "wrinkle"
]


def predict_mock(image):
    predicted_class = random.choice(CLASSES)
    confidence = round(random.uniform(0.70, 0.98), 2)

    return {
        "class": predicted_class,
        "confidence": confidence
    }