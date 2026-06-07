# SKINDO
## Sistema de Detección de Imperfecciones Faciales mediante Inteligencia Artificial

### Descripción General del Proyecto

**SkinDo** es una aplicación web desarrollada con Inteligencia Artificial capaz de analizar imágenes faciales y detectar diferentes imperfecciones visibles de la piel mediante técnicas de **Visión por Computador** y **Deep Learning**.

La aplicación permite al usuario capturar una fotografía de su rostro o cargar una imagen desde su dispositivo. Posteriormente, la imagen es procesada por un modelo de aprendizaje profundo entrenado para reconocer distintas condiciones de la piel. Finalmente, el sistema genera recomendaciones orientativas relacionadas con los resultados obtenidos.

> **Importante:** SkinDo es una herramienta de apoyo y no pretende reemplazar el diagnóstico ni la evaluación realizada por profesionales de la salud. Antes de una implementación real, la herramienta debería ser validada con el acompañamiento de dermatólogos u otros especialistas.

---

###  Objetivo General

Desarrollar una aplicación inteligente capaz de detectar diferentes imperfecciones faciales mediante análisis de imágenes y proporcionar recomendaciones generales para el cuidado de la piel.

---

### Objetivos Específicos

* Capturar imágenes faciales desde una cámara o mediante carga de archivos.
* Detectar automáticamente la presencia de un rostro válido.
* Clasificar diferentes condiciones visibles de la piel.
* Mostrar recomendaciones asociadas a cada condición detectada.
* Almacenar resultados e historial de análisis.
* Construir una interfaz accesible para usuarios sin conocimientos técnicos.

---

### Utilidad del Proyecto

La aplicación puede utilizarse como una herramienta de apoyo para:

* Monitoreo básico del estado de la piel.
* Clínicas de estética y centros de belleza como herramienta de primera consulta.
* Farmacias y tiendas cosméticas para recomendar productos según las necesidades del cliente.
* Seguimiento de tratamientos cosméticos.
* Investigación y aprendizaje en áreas relacionadas con Inteligencia Artificial y Visión por Computador.

---

###  Estructura del Repositorio

#### 📂 Notebooks modelos

Esta carpeta contiene todos los notebooks utilizados durante el desarrollo del modelo de Inteligencia Artificial.

Los notebooks incluyen el flujo completo de trabajo:

* Análisis Exploratorio de Datos (EDA).
* Limpieza y preprocesamiento de imágenes.
* Preparación y organización del dataset.
* Aumento de datos (*Data Augmentation*).
* Entrenamiento del modelo YOLOv8.
* Evaluación de resultados.
* Experimentos y ajustes de hiperparámetros.

Siguiendo los notebooks es posible reproducir todo el proceso de desarrollo del modelo, desde el análisis inicial de los datos hasta el entrenamiento final de YOLOv8.

---

#### 📂 Streamlit app

Esta carpeta contiene todos los archivos `.py` necesarios para desarrollar y ejecutar la aplicación web **SkinDo**.

Entre sus componentes se incluyen:

* Interfaz gráfica desarrollada con Streamlit.
* Carga de imágenes desde el dispositivo.
* Captura de imágenes mediante cámara.
* Integración del modelo YOLOv8 para inferencia.
* Generación de recomendaciones.
* Visualización de resultados.
* Gestión del historial de análisis.

Los archivos contienen las indicaciones necesarias para comprender el funcionamiento y despliegue de la aplicación.

---

### Tecnologías Utilizadas

* Python
* Streamlit
* YOLOv8
* Ultralytics
* OpenCV
* NumPy
* Pandas
* Pillow
* Deep Learning
* Computer Vision

---

### Licencia

Este proyecto ha sido desarrollado con fines educativos, de investigación y demostración tecnológica.

Antes de utilizarlo en entornos reales o comerciales, se recomienda realizar validaciones adicionales y verificar las licencias de los datasets y herramientas utilizadas.

---

###  Autores

Proyecto desarrollado por Carlos Reales y Leslie Mosquera como parte de un trabajo académico enfocado en la aplicación de técnicas de Inteligencia Artificial y Visión por Computador para el análisis automático de la piel.
