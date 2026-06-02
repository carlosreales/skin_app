import streamlit as st
from PIL import Image
import os
from datetime import datetime
import pandas as pd


from cloud_storage import upload_image_to_cloudinary, delete_image_from_cloudinary
from image_utils import draw_detections
from predictor import predict
from recommendations import RECOMMENDATIONS
from database import create_tables, save_analysis, get_analysis_history, delete_all_analysis
from face_detector import analyze_face_position
from offline_recommendations import OFFLINE_RECOMMENDATIONS
from ai_recommendations import get_ai_recommendations


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
    page_title="SkinDo",
    page_icon="🧴",
    layout="centered"
)

create_tables()

st.title("SkinDo")
st.write("Aplicación demo para detectar imperfecciones faciales en imágenes.")

st.warning(
    "Esta herramienta es solo orientativa y no reemplaza una valoración médica o dermatológica."
)

page = st.sidebar.radio(
    "Menú",
    [
        "Analizar imagen",
        "Guía de recomendaciones",
        "Historial",
        "Estadísticas"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Configuración del modelo")

conf_threshold = st.sidebar.slider(
    "Confianza mínima",
    min_value=0.01,
    max_value=0.90,
    value=0.10,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "Umbral IoU",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)


if page == "Analizar imagen":

    option = st.radio(
        "Selecciona una opción:",
        ["Tomar foto con cámara", "Subir imagen"]
    )

    image = None

    if option == "Tomar foto con cámara":

        st.markdown(
            """
            <style>
                div[data-testid="stCameraInput"] {
                    position: relative;
                }

                div[data-testid="stCameraInput"]::after {
                    content: "👤";
                    font-size: 3rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: rgba(255, 255, 255, 0.25);

                    position: absolute;
                    top: 46%;
                    left: 50%;
                    transform: translate(-50%, -50%);

                    width: 210px;
                    height: 270px;

                    border: 3px dashed #4A7C59;
                    border-radius: 50%;
                    z-index: 99;
                    pointer-events: none;

                    animation: pulse-guide 2s ease-in-out infinite;
                    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.25);
                }

                @keyframes pulse-guide {
                    0% { opacity: 0.45; }
                    50% { opacity: 0.9; }
                    100% { opacity: 0.45; }
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "📷 Centra tu rostro dentro del óvalo, con buena iluminación frontal y sin cubrir ojos, nariz ni boca."
        )

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
        image = image.convert("RGB")

        st.image(
            image,
            caption="Imagen seleccionada",
            use_container_width=True
        )

        if st.button("Analizar imagen"):

            face_analysis = analyze_face_position(image)

            st.subheader("Validación de captura")

            st.image(
                face_analysis["annotated_image"],
                caption="Rostro detectado por OpenCV",
                use_container_width=True
            )

            if not face_analysis["face_detected"]:
                st.error(face_analysis["message"])

            else:
                if face_analysis["quality"] == "Buena":
                    st.success(face_analysis["message"])
                elif face_analysis["quality"] == "Aceptable":
                    st.warning(face_analysis["message"])
                else:
                    st.error(face_analysis["message"])

                os.makedirs("uploads", exist_ok=True)

                image_filename = (
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".jpg"
                )

                image_path = os.path.join(
                    "uploads",
                    image_filename
                )

                image.save(
                    image_path,
                    "JPEG"
                )

                result = predict(
                    image,
                    conf_threshold,
                    iou_threshold
                )

                predicted_class = result["class"]
                confidence = result["confidence"]
                detections = result.get("detections", [])

                severity = get_severity(confidence)

                description = DESCRIPTIONS.get(
                    predicted_class,
                    "No hay descripción disponible."
                )

                annotated_image = None

                valid_detections = []

                for detection in detections:
                    if "box" in detection:
                        valid_detections.append(detection)

                if len(valid_detections) > 0:
                    annotated_image = draw_detections(
                        image,
                        valid_detections
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

                st.subheader("Imagen analizada por YOLO")

                if annotated_image is not None:
                    col_original, col_annotated = st.columns(2)

                    with col_original:
                        st.write("Imagen original")
                        st.image(
                            image,
                            use_container_width=True
                        )

                    with col_annotated:
                        st.write("Imagen con detecciones")
                        st.image(
                            annotated_image,
                            use_container_width=True
                        )

                else:
                    st.info("No hay detecciones con coordenadas válidas para dibujar.")

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

                st.subheader("Recomendaciones inteligentes")

                ai_recommendation = get_ai_recommendations(
                    predicted_class,
                    display_name
                )

                if ai_recommendation is not None:
                    st.info(ai_recommendation.get("description", ""))

                    st.write("### Rutina sugerida")

                    for step in ai_recommendation.get("routine", []):
                        st.write(f"✅ {step}")

                    st.write("### Productos sugeridos")

                    for product in ai_recommendation.get("products", []):
                        st.write(f"**{product.get('name', '')}**")
                        st.write(f"Marca: {product.get('brand', '')}")
                        st.write(f"Tipo: {product.get('type', '')}")
                        st.write(f"Precio aproximado: {product.get('price', '')}")
                        st.write(f"Uso: {product.get('how_to_use', '')}")

                        if product.get("warning"):
                            st.warning(product.get("warning"))

                        st.divider()

                    st.warning(
                        ai_recommendation.get(
                            "medical_note",
                            "Estas recomendaciones son orientativas y no reemplazan una valoración dermatológica."
                        )
                    )

                else:
                    st.write("Usando recomendaciones offline:")

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

if page == "Guía de recomendaciones":

    st.subheader("Guía offline de recomendaciones")

    st.write(
        "Consulta cuidados y productos sugeridos para cada tipo de imperfección detectada por el modelo."
    )

    selected_condition = st.selectbox(
        "Selecciona una condición",
        list(OFFLINE_RECOMMENDATIONS.keys()),
        format_func=get_display_name
    )

    recommendation = OFFLINE_RECOMMENDATIONS[selected_condition]

    st.divider()

    st.header(recommendation["title"])

    st.info(recommendation["description"])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Cuidados recomendados")

        for item in recommendation["care"]:
            st.write(f"✅ {item}")

    with col2:
        st.subheader("Producto de ejemplo")

        for product in recommendation["products"]:

            st.image(
                product["image"],
                caption=product["name"],
                use_container_width=True
            )

            st.markdown(f"### {product['name']}")
            st.write(f"**Marca:** {product['brand']}")
            st.write(f"**Tipo:** {product['type']}")
            st.write(f"**Precio aproximado:** {product['price']}")

            st.link_button(
                "Ver producto",
                product["url"]
            )

    st.warning(
        "Estas recomendaciones son orientativas. Los precios pueden cambiar y no reemplazan una valoración médica o dermatológica."
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

            st.success(
                "Historial, imágenes locales e imágenes de Cloudinary eliminados correctamente."
            )

            st.rerun()

        else:
            st.error("Debes marcar la confirmación antes de eliminar.")

    st.divider()

    history = get_analysis_history()

    if len(history) == 0:
        st.write("Todavía no hay análisis guardados.")

    else:
        for row in history:
            (
                analysis_id,
                analysis_date,
                predicted_class,
                confidence,
                image_path,
                severity,
                description,
                cloudinary_public_id
            ) = row

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

        df_display["predicted_class"] = df_display["predicted_class"].apply(
            get_display_name
        )

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