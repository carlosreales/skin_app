# SkinDo

## Descripción del Proyecto

SkinDo es una aplicación web desarrollada en Python y Streamlit para el análisis de imágenes faciales mediante técnicas de Procesamiento de Imágenes e Inteligencia Artificial.

La aplicación permite capturar o cargar fotografías del rostro, validar la presencia y calidad de la captura facial mediante OpenCV y detectar imperfecciones cutáneas utilizando un modelo YOLO entrenado previamente.

Además, el sistema ofrece recomendaciones de cuidado facial, historial de análisis, estadísticas básicas y almacenamiento seguro de imágenes en la nube.

---

# Objetivos

## Objetivo General

Desarrollar una aplicación web capaz de detectar imperfecciones faciales utilizando técnicas de visión por computador e inteligencia artificial.

## Objetivos Específicos

* Detectar automáticamente la presencia de un rostro en una imagen.
* Analizar imperfecciones faciales mediante un modelo YOLO.
* Visualizar las zonas detectadas sobre la imagen.
* Almacenar resultados de análisis para futuras consultas.
* Generar recomendaciones de cuidado facial.
* Presentar estadísticas básicas de uso y detección.

---

# Tecnologías Utilizadas

## Backend

* Python 3.x
* Streamlit

## Procesamiento de Imágenes

* OpenCV
* Pillow (PIL)

## Inteligencia Artificial

* Ultralytics YOLO
* Modelo personalizado: `dermavision_best_v3.pt`

## Base de Datos

* SQLite

## Almacenamiento en la Nube

* Cloudinary

## Control de Versiones

* Git
* GitHub

## IA Generativa (Experimental)

* Groq API
* Llama 3

---

# Arquitectura General

Usuario

↓

Captura o carga imagen

↓

Validación facial (OpenCV Haar Cascade)

↓

Detección de imperfecciones (YOLO)

↓

Visualización de resultados

↓

Almacenamiento

* SQLite
* Cloudinary

↓

Recomendaciones

* Offline
* IA (Groq)

↓

Historial y estadísticas

---

# Clases Detectadas por el Modelo

El modelo actual permite detectar las siguientes categorías:

| ID | Clase       |
| -- | ----------- |
| 0  | Acne        |
| 1  | Dark Circle |
| 2  | Darkspot    |
| 3  | Dry         |
| 4  | Normal Skin |
| 5  | Oily        |
| 6  | Wrinkle     |

---

# Funcionalidades Implementadas

## 1. Captura de Imágenes

La aplicación permite:

* Tomar fotografías mediante la cámara del dispositivo.
* Subir imágenes existentes.
* Soporte para formatos:

  * JPG
  * JPEG
  * PNG

---

## 2. Guía Visual para Captura

Se implementó una guía visual sobre el componente de cámara para ayudar al usuario a posicionar correctamente el rostro antes de realizar la captura.

---

## 3. Validación de Rostro

Antes de ejecutar el modelo YOLO:

* Se verifica la presencia de un rostro.
* Se analiza la calidad de la captura.
* Se valida el centrado del rostro.

Tecnología utilizada:

* OpenCV Haar Cascade

---

## 4. Detección de Imperfecciones

La detección se realiza mediante un modelo YOLO personalizado.

Características:

* Detección de múltiples zonas.
* Selección automática de la detección con mayor confianza.
* Visualización mediante bounding boxes.
* Visualización del porcentaje de confianza.

---

## 5. Configuración del Modelo

El usuario puede configurar:

### Umbral de Confianza

Permite definir la confianza mínima requerida para mostrar detecciones.

Valor inicial:

0.10

Nota:

Debido a las limitaciones actuales del modelo, valores superiores a 0.50 reducen considerablemente la cantidad de detecciones obtenidas.

### Umbral IoU

Permite controlar el nivel de solapamiento permitido entre detecciones.

Valor recomendado:

0.45

---

## 6. Recomendaciones Offline

La aplicación incluye una guía integrada de cuidado facial organizada por tipo de imperfección.

Cada categoría incluye:

* Descripción
* Cuidados sugeridos
* Productos recomendados
* Imagen del producto
* Precio aproximado
* Enlace de compra

---

## 7. Recomendaciones Inteligentes (Experimental)

Se encuentra en desarrollo la integración con Groq y modelos LLM para generar:

* Rutinas personalizadas
* Consejos de cuidado facial
* Explicaciones contextualizadas

---

## 8. Historial de Análisis

Se almacena información de cada análisis:

* Fecha
* Resultado
* Confianza
* Severidad
* Imagen asociada

El historial permite:

* Consultar análisis anteriores
* Visualizar imágenes almacenadas
* Eliminar registros

---

## 9. Estadísticas

La aplicación incluye un módulo estadístico básico que muestra:

* Total de análisis realizados
* Confianza promedio
* Clase más frecuente
* Distribución de clases
* Distribución por severidad

---

## 10. Almacenamiento

### SQLite

Almacena:

* Resultados
* Fechas
* Confianza
* Descripciones
* Referencias de imágenes

### Cloudinary

Almacena:

* Imágenes originales
* Recursos asociados al análisis

Ventajas:

* Persistencia de imágenes
* Acceso desde Streamlit Cloud
* Eliminación remota de archivos

---

# Estructura del Proyecto

```text
skin_app/
│
├── app.py
├── predictor.py
├── face_detector.py
├── image_utils.py
├── recommendations.py
├── offline_recommendations.py
├── ai_recommendations.py
├── cloud_storage.py
├── database.py
│
├── models/
│   └── dermavision_best_v3.pt
│
├── assets/
│   └── products/
│
├── uploads/
│
├── skin_analysis.db
│
├── requirements.txt
└── README.md
```

# Próximas Mejoras

* Exportación de informes PDF.
* Dashboard avanzado.
* Integración completa con Groq.
* Migración de SQLite a Supabase.
* Mejora de la precisión del modelo.
* Comparación de análisis históricos.
* Sistema de autenticación de usuarios.

# Autor

Proyecto académico desarrollado para la asignatura de Procesamiento de Imágenes.

Tecnologías principales:

Python • Streamlit • OpenCV • YOLO • SQLite • Cloudinary • Groq
