import streamlit as st
from PIL import Image
import os
from datetime import datetime
import pandas as pd

from predictor import predict_mock
from recommendations import RECOMMENDATIONS
from database import create_tables, save_analysis, get_analysis_history
from face_detector import detect_face


st.set_page_config(
    page_title="Skin AI Demo",
    page_icon="🧴",
    layout="centered"
)

create_tables()

st.title("Skin AI Demo")
st.write("Analiza una imagen del rostro y entrega recomendaciones generales según la condición detectada.")

st.warning(
    "Esta herramienta es solo orientativa y no reemplaza una valoración médica o dermatológica."
)

page = st.sidebar.radio(
    "Menú",
    [
        "Analizar imagen",
        "Historial",
        "Estadísticas"
    ]
)


if page == "Analizar imagen":

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

            if not detect_face(image):
                st.error("No se detectó un rostro en la imagen.")

            else:
                result = predict_mock(image)

                predicted_class = result["class"]
                confidence = result["confidence"]

                os.makedirs("uploads", exist_ok=True)

                image_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                image_path = os.path.join("uploads", image_filename)

                image.save(image_path)

                save_analysis(predicted_class, confidence, image_path)

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


if page == "Historial":

    st.subheader("Historial de análisis")

    history = get_analysis_history()

    if len(history) == 0:
        st.write("Todavía no hay análisis guardados.")
    else:
        for row in history:
            analysis_id, analysis_date, predicted_class, confidence, image_path = row

            st.write(
                f"**#{analysis_id}** | {analysis_date} | "
                f"{predicted_class} | {confidence * 100:.0f}%"
            )

            if image_path:
                st.image(image_path, width=150)

            st.divider()


if page == "Estadísticas":

    st.subheader("Estadísticas generales")

    history = get_analysis_history()

    if len(history) == 0:
        st.write("No hay datos suficientes para mostrar estadísticas.")
    else:
        df = pd.DataFrame(
            history,
            columns=[
                "id",
                "analysis_date",
                "predicted_class",
                "confidence",
                "image_path"
            ]
        )

        total_analysis = len(df)
        avg_confidence = df["confidence"].mean()
        most_common_class = df["predicted_class"].mode()[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Total análisis", total_analysis)
        col2.metric("Confianza promedio", f"{avg_confidence * 100:.0f}%")
        col3.metric("Clase más frecuente", most_common_class)

        st.subheader("Distribución de clases detectadas")

        class_counts = df["predicted_class"].value_counts()

        st.bar_chart(class_counts)