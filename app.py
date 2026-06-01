import streamlit as st
from PIL import Image
import os
from datetime import datetime
import pandas as pd

from predictor import predict
from recommendations import RECOMMENDATIONS
from database import create_tables, save_analysis, get_analysis_history, delete_all_analysis
from face_detector import detect_face


DESCRIPTIONS = {
    "acne": "Se detectaron características compatibles con acné facial.",
    "dark circle": "Se detectaron ojeras visibles alrededor de los ojos.",
    "darkspot": "Se identificaron manchas o hiperpigmentación facial.",
    "dry": "La piel presenta signos visibles de resequedad.",
    "normal skin": "La piel presenta características compatibles con una piel saludable y equilibrada.",
    "oily": "Se detectaron características asociadas a piel grasa.",
    "wrinkle": "Se identificaron líneas de expresión o arrugas."
}


def get_severity(confidence):
    if confidence < 0.50:
        return "Baja"
    elif confidence < 0.80:
        return "Media"
    else:
        return "Alta"
    
def delete_uploads_folder():
    if os.path.exists("uploads"):
        for filename in os.listdir("uploads"):
            file_path = os.path.join("uploads", filename)

            if os.path.isfile(file_path):
                os.remove(file_path)


st.set_page_config(
    page_title="SkinDo",
    page_icon="🧴",
    layout="centered"
)

create_tables()

st.title("SkinDo")
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
                result = predict(image)

                predicted_class = result["class"]
                confidence = result["confidence"]
                detections = result.get("detections", [])

                st.subheader("Detecciones del modelo")

                if len(detections) == 0:
                    st.warning(
                        "El modelo no detectó ninguna imperfección con el umbral actual."
                    )
                else:
                    for detection in detections:
                        st.write(
                            f"**{detection['class']}** - "
                            f"{detection['confidence'] * 100:.0f}%"
                        )
                        st.progress(detection["confidence"])

                severity = get_severity(confidence)

                description = DESCRIPTIONS.get(
                    predicted_class,
                    "No hay descripción disponible."
                )

                os.makedirs("uploads", exist_ok=True)

                image_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                image_path = os.path.join("uploads", image_filename)

                image_to_save = image.convert("RGB")
                image_to_save.save(image_path, format="JPEG")

                image.save(image_path)

                save_analysis(
                    predicted_class,
                    confidence,
                    image_path,
                    severity,
                    description
                )

                st.divider()

                if predicted_class == "normal skin":
                    st.success("🟢 Piel Normal Detectada")

                elif predicted_class == "acne":
                    st.warning("🟠 Acné Detectado")

                elif predicted_class == "dark circle":
                    st.warning("🟤 Ojeras Detectadas")

                elif predicted_class == "darkspot":
                    st.warning("🟤 Manchas Detectadas")

                elif predicted_class == "dry":
                    st.warning("🟡 Piel Seca Detectada")

                elif predicted_class == "oily":
                    st.warning("🟠 Piel Grasa Detectada")

                elif predicted_class == "wrinkle":
                    st.info("🔵 Arrugas Detectadas")

                else:
                    st.info(f"Resultado: {predicted_class}")

                col1, col2 = st.columns(2)

                col1.metric("Confianza", f"{confidence * 100:.0f}%")
                col2.metric("Severidad", severity)

                st.write("Nivel de confianza")
                st.progress(float(confidence))

                st.subheader("Interpretación")
                st.info(description)

                st.subheader("Recomendaciones")

                recommendations = RECOMMENDATIONS.get(
                    predicted_class,
                    []
                )

                for item in recommendations:
                    st.write(f"✅ {item}")

                st.warning(
                    "Esta información es orientativa y no reemplaza una valoración médica profesional."
                )


if page == "Historial":

    st.subheader("Historial de análisis")

    st.warning("Esta acción eliminará todos los análisis guardados y las imágenes locales.")

    confirm_delete = st.checkbox("Confirmo que deseo eliminar todo el historial")

    if st.button("Eliminar historial completo"):
        if confirm_delete:
            delete_all_analysis()
            delete_uploads_folder()
            st.success("Historial e imágenes eliminados correctamente.")
            st.rerun()
        else:
            st.error("Debes marcar la confirmación antes de eliminar.")

    history = get_analysis_history()

    if len(history) == 0:
        st.write("Todavía no hay análisis guardados.")
    else:
        for row in history:
            analysis_id, analysis_date, predicted_class, confidence, image_path, severity, description = row

            st.write(f"### Análisis #{analysis_id}")
            st.write(f"**Fecha:** {analysis_date}")

            col1, col2 = st.columns([1, 2])

            with col1:
                if image_path and os.path.exists(image_path):
                    st.image(image_path, width=160)
                else:
                    st.warning("Imagen no disponible.")

            with col2:
                st.write(f"**Condición detectada:** {predicted_class}")
                st.write(f"**Confianza:** {confidence * 100:.0f}%")
                st.write(f"**Severidad:** {severity}")
                st.write(f"**Interpretación:** {description}")

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
                "image_path",
                "severity",
                "description"
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

        st.subheader("Distribución por severidad")
        severity_counts = df["severity"].value_counts()
        st.bar_chart(severity_counts)

        st.subheader("Tabla de análisis")

        st.dataframe(
            df[
                [
                    "id",
                    "analysis_date",
                    "predicted_class",
                    "confidence",
                    "severity"
                ]
            ],
            use_container_width=True
        )