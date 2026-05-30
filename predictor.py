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


USE_REAL_MODEL = False


def predict_mock(image):
    predicted_class = random.choice(CLASSES)
    confidence = round(random.uniform(0.70, 0.98), 2)

    return {
        "class": predicted_class,
        "confidence": confidence
    }


def predict_real_model(image):
    # Aquí conectaremos el modelo real cuando esté disponible.
    # Por ahora dejamos esta función preparada.
    return predict_mock(image)


def predict(image):
    if USE_REAL_MODEL:
        return predict_real_model(image)
    else:
        return predict_mock(image)