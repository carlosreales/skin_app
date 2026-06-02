import streamlit as st
from PIL import Image
import os
from datetime import datetime
import pandas as pd
from cloud_storage import upload_image_to_cloudinary, delete_image_from_cloudinary

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
    "wrinkle": "Se identificaron líneas de expresión o arrugas.",
    "Sin detección": "El modelo no detectó imperfecciones con el umbral de confianza actual."
}


def get_severity(confidence):
    if confidence == 0:
        return "No aplica"
    elif confidence < 0.50:
        return "Baja"
    elif confidence < 0.80:
        return "Media"
    else:
        return "Alta"


def get_display_name(predicted_class):
    names = {
        "acne": "Acné",
        "dark circle": "Ojeras",
        "darkspot": "Manchas",
        "dry": "Piel seca",
        "normal skin": "Piel normal",
        "oily": "Piel grasa",
        "wrinkle": "Arrugas",
        "Sin detección": "Sin detección"
    }

    return names.get(predicted_class, predicted_class)


def delete_uploads_folder():
    if os.path.exists("uploads"):
        for filename in os.listdir("uploads"):
            file_path = os.path.join("uploads", filename)

            if os.path.isfile(file_path):
                os.remove(file_path)


st.set_page_config(
    page_title="Skin AI Demo",
    page_icon="🧴",
    layout="centered"
)

create_tables()

st.title("Skin AI Demo")
st.write("Aplicación demo para detectar imperfecciones faciales en imágenes.")

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

            image = image.convert("RGB")

            if not detect_face(image):
                st.error("No se detectó un rostro en la imagen.")

            else:
                result = predict(image)

                predicted_class = result["class"]
                confidence = result["confidence"]
                detections = result.get("detections", [])

                severity = get_severity(confidence)

                description = DESCRIPTIONS.get(
                    predicted_class,
                    "No hay descripción disponible."
                )

                os.makedirs("uploads", exist_ok=True)

                image_filename = (
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".jpg"
                )

                image_path = os.path.join(
                    "uploads",
                    image_filename
                )

                image_to_save = image.convert("RGB")

                image_to_save.save(
                    image_path,
                    "JPEG"
                )

                cloudinary_result = upload_image_to_cloudinary(image_path)

                cloud_image_url = cloudinary_result["secure_url"]
                cloudinary_public_id = cloudinary_result["public_id"]

                save_analysis(
                    predicted_class,
                    confidence,
                    cloud_image_url,
                    severity,
                    description,
                    cloudinary_public_id
                )

                st.divider()

                display_name = get_display_name(predicted_class)

                if predicted_class == "Sin detección":
                    st.warning("⚪ El modelo no detectó imperfecciones relevantes")

                elif predicted_class == "normal skin":
                    st.success("🟢 Resultado principal: Piel normal")

                elif predicted_class == "acne":
                    st.warning("🟠 Resultado principal: Acné")

                elif predicted_class == "dark circle":
                    st.warning("🟤 Resultado principal: Ojeras")

                elif predicted_class == "darkspot":
                    st.warning("🟤 Resultado principal: Manchas")

                elif predicted_class == "dry":
                    st.warning("🟡 Resultado principal: Piel seca")

                elif predicted_class == "oily":
                    st.warning("🟠 Resultado principal: Piel grasa")

                elif predicted_class == "wrinkle":
                    st.info("🔵 Resultado principal: Arrugas")

                else:
                    st.info(f"Resultado principal: {display_name}")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Resultado",
                    display_name
                )

                col2.metric(
                    "Confianza",
                    f"{confidence * 100:.0f}%"
                )

                col3.metric(
                    "Severidad",
                    severity
                )

                st.write("Nivel de confianza del resultado principal")
                st.progress(float(confidence))

                st.subheader("Interpretación")
                st.info(description)

                st.subheader("Detecciones del modelo")

                if len(detections) == 0:
                    st.warning(
                        "El modelo no detectó imperfecciones con el umbral actual."
                    )
                else:
                    st.write(
                        f"Se detectaron **{len(detections)} zona(s)** en la imagen."
                    )

                    detection_summary = {}

                    for detection in detections:
                        class_name = detection["class"]

                        if class_name not in detection_summary:
                            detection_summary[class_name] = 0

                        detection_summary[class_name] += 1

                    st.write("Resumen por clase detectada:")

                    for class_name, total in detection_summary.items():
                        st.write(
                            f"- **{get_display_name(class_name)}:** {total} detección(es)"
                        )

                    st.write("Detalle de detecciones:")

                    for index, detection in enumerate(detections, start=1):
                        detection_name = get_display_name(detection["class"])
                        detection_confidence = detection["confidence"]

                        st.write(
                            f"**Zona {index}:** {detection_name} - "
                            f"{detection_confidence * 100:.0f}%"
                        )

                        st.progress(float(detection_confidence))

                st.subheader("Recomendaciones")

                recommendations = RECOMMENDATIONS.get(
                    predicted_class,
                    []
                )

                if len(recommendations) == 0:
                    st.write("No hay recomendaciones disponibles para este resultado.")
                else:
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

            history_to_delete = get_analysis_history()

            for row in history_to_delete:
                cloudinary_public_id = row[7]

                if cloudinary_public_id:
                    delete_image_from_cloudinary(cloudinary_public_id)

            delete_all_analysis()
            delete_uploads_folder()

            st.success("Historial, imágenes locales e imágenes de Cloudinary eliminados correctamente.")
            st.rerun()

        else:
            st.error("Debes marcar la confirmación antes de eliminar.")

    st.divider()

    history = get_analysis_history()

    if len(history) == 0:
        st.write("Todavía no hay análisis guardados.")
    else:
        for row in history:
            analysis_id, analysis_date, predicted_class, confidence, image_path, severity, description, cloudinary_public_id = row

            st.write(f"### Análisis #{analysis_id}")
            st.write(f"**Fecha:** {analysis_date}")

            col1, col2 = st.columns([1, 2])

            with col1:
                if image_path:
                    st.image(image_path, width=160)
                else:
                    st.warning("Imagen no disponible.")

            with col2:
                st.write(f"**Resultado principal:** {get_display_name(predicted_class)}")
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
                "description",
                "cloudinary_public_id"
            ]
        )

        total_analysis = len(df)
        avg_confidence = df["confidence"].mean()
        most_common_class = df["predicted_class"].mode()[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Total análisis", total_analysis)
        col2.metric("Confianza promedio", f"{avg_confidence * 100:.0f}%")
        col3.metric("Clase más frecuente", get_display_name(most_common_class))

        st.subheader("Distribución de resultados principales")

        class_counts = df["predicted_class"].apply(get_display_name).value_counts()

        st.bar_chart(class_counts)

        st.subheader("Distribución por severidad")

        severity_counts = df["severity"].value_counts()

        st.bar_chart(severity_counts)

        st.subheader("Tabla de análisis")

        df_display = df.copy()
        df_display["predicted_class"] = df_display["predicted_class"].apply(get_display_name)
        df_display["confidence"] = df_display["confidence"].apply(
            lambda x: f"{x * 100:.0f}%"
        )

        st.dataframe(
            df_display[
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