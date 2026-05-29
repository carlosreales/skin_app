import streamlit as st
from PIL import Image

from predictor import predict_mock
from recommendations import RECOMMENDATIONS


st.set_page_config(
    page_title="Skin AI Demo",
    page_icon="🧴",
    layout="centered"
)

st.title("Skin AI Demo")
st.write("Analiza una imagen del rostro y entrega recomendaciones generales según la condición detectada.")

st.warning(
    "Esta herramienta es solo orientativa y no reemplaza una valoración médica o dermatológica."
)

option = st.radio(
    "Selecciona una opción:",
    ["Tomar foto con cámara", "Subir imagen"]
)

image = None

if option == "Tomar foto con cámara":
    camera_image = st.camera_input("Toma una foto de tu rostro")
    if camera_image is not None:
        image = Image.open(camera_image)

else:
    uploaded_file = st.file_uploader(
        "Sube una imagen del rostro",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)


if image is not None:
    st.image(image, caption="Imagen seleccionada", use_container_width=True)

    if st.button("Analizar imagen"):
        result = predict_mock(image)

        predicted_class = result["class"]
        confidence = result["confidence"]

        st.subheader("Resultado")
        st.write(f"**Condición detectada:** {predicted_class}")
        st.write(f"**Confianza:** {confidence * 100:.0f}%")

        st.subheader("Recomendaciones")

        recommendations = RECOMMENDATIONS.get(predicted_class, [])

        for item in recommendations:
            st.write(f"- {item}")

        st.info(
            "Para una evaluación precisa, consulta con un dermatólogo o profesional de la salud."
        )