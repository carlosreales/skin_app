# SKINDO
Sistema de identificación de afecciones cutáneas faciales y recomendación de productos cosméticos mediante visión artificial.

El sistema detecta 7 condiciones de piel a partir de una imagen o cámara web y genera recomendaciones personalizadas de productos.

> Este sistema es orientativo y no reemplaza la consulta médica profesional.

---

## Tabla de contenidos

- [Requisitos del sistema](#requisitos-del-sistema)
- [Instalación del entorno](#instalación-del-entorno)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Ejecución paso a paso](#ejecución-paso-a-paso)
  - [Notebook 01 — EDA](#notebook-01--eda)
  - [Notebook 02 — Preparación del dataset](#notebook-02--preparación-del-dataset)
  - [Notebook 03 — Viola Jones](#notebook-03--violajones-training)
  - [Notebook 04 — Entrenamiento en Kaggle](#notebook-04--entrenamiento-en-kaggle)
  - [App Streamlit — Demo](#app-streamlit--demo)
- [Decisiones tomadas durante el proyecto](#decisiones-tomadas-durante-el-proyecto)
- [Clases detectadas](#clases-detectadas)
- [Datasets utilizados](#datasets-utilizados)
- [Limitaciones]
- [Líneas Futuras de Trabajo]
---

## Requisitos del sistema

- Python 3.10 o superior
- Windows 10/11, macOS 12+ o Ubuntu 20.04+
- Mínimo 8 GB de RAM
- Mínimo 15 GB de espacio en disco (datasets + modelo)
- Cuenta gratuita en [Roboflow](https://roboflow.com) para la descarga de datasets
- Cuenta en [Kaggle](https://www.kaggle.com) para el entrenamiento con GPU T4

---

## Instalación del entorno

```bash
# 1. Clonar o descomprimir el proyecto
cd "Trabajo final"

# 2. Crear entorno virtual
python -m venv cv_env

# 3. Activar el entorno
# Windows:
cv_env\Scripts\activate
# macOS / Linux:
source cv_env/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

Crear el archivo `.env` en la raíz del proyecto (copiar desde `.env.example`):

```
ROBOFLOW_API_KEY=tu_api_key_de_roboflow
```

---

## Estructura del proyecto

```
Trabajo final/
├── app/
│   └── streamlit_app.py              # Interfaz de la demo
├── models/
│   └── dermavision_best_v3.pt        # Modelo final (YOLOv8m, 52 MB)
├── data/
│   ├── clean/                        # Dataset unificado (generado por nb02)
│   │   ├── train/images/
│   │   ├── train/labels/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml
│   ├── celeba/                       # Descargar manualmente (ver nb02)
│   │   ├── img_align_celeba/
│   │   │   └── img_align_celeba/    # Imagenes .jpg aqui
│   │   └── list_attr_celeba.txt
│   ├── face_skin_condition/          # Descargado via Roboflow API
│   ├── dark_circle_r/                # Descargado manualmente de Roboflow
│   ├── darkspot_rf/                  # Descargado via Roboflow API
│   
├── results/
│   └── reports/                      # Graficas generadas por los notebooks
├── 01_EDA.ipynb
├── 02_data_preparation_v2.ipynb
├── 03_violajones_training.ipynb      
├── 04_kaggle-training.ipynb          # Notebook ejecutado en Kaggle
├── config.yaml
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

---

## Ejecución paso a paso

### Notebook 01 — EDA

Abrir `01_EDA.ipynb` en VS Code o Jupyter y ejecutar todas las celdas.

El notebook descarga el dataset principal desde Roboflow y realiza el análisis exploratorio completo. Los hallazgos que condicionaron el resto del proyecto:

- El dataset tiene **4.717 imágenes** y **29.275 anotaciones** en 9 clases, todas en formato 640×640 píxeles
- **Desbalance crítico**: `acne` domina ampliamente mientras que `normal skin`, `skin redness` y `darkspot` son clases con muy pocas muestras
- El **48.8% de las bounding boxes** tienen área menor al 0.5% de la imagen, indicando lesiones muy pequeñas difíciles de detectar
- Se identificaron imágenes borrosas (varianza Laplaciana < 100) e imágenes repetidas que requieren tratamiento

Los reportes se guardan automáticamente en `results/reports/`.

---

### Notebook 02 — Preparación del dataset

Abrir `02_data_preparation_v2.ipynb` y ejecutar en orden. 

#### Prerequisitos antes de ejecutar

**CelebA** — descargar manualmente y colocar en esta estructura exacta:

```
data/celeba/
    img_align_celeba/
        img_align_celeba/     <- carpeta anidada, las imagenes van aqui
            000001.jpg
            000002.jpg
            ...
    list_attr_celeba.txt
```

Descarga en [Kaggle — CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

La ruta tiene la carpeta `img_align_celeba` anidada dos veces. Si tu descarga tiene una sola carpeta, ajusta la variable `celeba_imgs_dir` en el notebook.

**Datasets de Roboflow** — algunos via API y otros requieren descarga manual:

| Dataset | Metodo | Ruta destino |
|---------|--------|-------------|
| face_skin_condition | API automatica | `data/face_skin_condition/` |
| dark_circle | Manual desde Roboflow web | `data/dark_circle_r/` |
| darkspot | API automatica | `data/darkspot_rf/` |
| redness | Manual desde Roboflow web | `data/redness/` |

Para los manuales: ir a la URL del dataset en Roboflow, seleccionar formato YOLOv8, descargar ZIP y extraer en la carpeta indicada.

#### Que hace el notebook

**Seccion 2 — Integracion de CelebA**

Filtra las 202.599 imágenes de CelebA por atributos para generar dos clases:
- `normal_skin`: imágenes con `Bags_Under_Eyes = -1` — se generan ~21.500 imágenes
- `dark_circle`: imágenes con `Bags_Under_Eyes = 1`, sin sombrero ni gafas — se generan ~15.000 imágenes

Para cada imagen: Viola-Jones detecta el rostro, recorta con padding de 40 px y redimensiona a 640×640. Las bounding boxes se generan geométricamente.

**Secciones 3-5 — Integracion de datasets Roboflow**

Cada fuente tiene un mapeo de clases. La clase `acne` se omite en todos los externos porque el dataset original ya tiene volumen suficiente.

**Seccion 7 — Deteccion de duplicados**

- Hash MD5 exacto: detecta archivos idénticos a nivel de bytes
- Similitud de histograma HSV con correlación >= 0.98: detecta duplicados aproximados

Los duplicados aproximados no se eliminaron porque al visualizar los pares detectados no coincidían correctamente. Las imágenes sin rostro tampoco se eliminan aquí: serán filtradas por Viola-Jones en el entrenamiento.

**Seccion 8 — Mejora de calidad**

Solo se mejoran imágenes con varianza Laplaciana < 50. El umbral original de 100 se redujo a 50 para no modificar imágenes aceptables. Pipeline:

1. `cv2.medianBlur(ksize=3)` — mas rapido que NLM para este volumen
2. Unsharp Mask (alpha=1.4)
3. CLAHE en canal L del espacio LAB (clipLimit=2.0)

El dataset final en `data/clean/` tiene aproximadamente **44.000 imágenes** en train. El desbalance persiste — se corrige en el notebook de entrenamiento.

---
### Notebook 03 — Viola Jones


---
### Notebook 04 — Entrenamiento en Kaggle

El entrenamiento se realizó en Kaggle con GPU Tesla T4 (15.6 GB VRAM). No es reproducible en local con CPU en tiempo razonable.

#### Subir el dataset a Kaggle

1. Comprimir `data/clean/` en un ZIP
2. Ir a [kaggle.com/datasets](https://www.kaggle.com/datasets) → New Dataset
3. Subir el ZIP con nombre `clean_vj`
4. En el notebook, el dataset estará en `/kaggle/input/datasets/lesliemosquera/clean-vj/clean_vj/`

#### Configurar el entorno en Kaggle

1. Crear nuevo notebook y subir `04_kaggle-training.ipynb`
2. Panel derecho → Add Data → añadir `clean_vj`
3. Panel derecho → Session Options → Accelerator → **GPU T4**
4. Ejecutar todas las celdas en orden

#### Que hace el notebook

**Balanceo del dataset**

Con 44k imágenes el dataset seguía desbalanceado (normal_skin: 22k, dark_circle: 18k, acne: 10k). Se aplicó:

- Eliminación de `pores` (440 anotaciones) y `skin_redness` (376 anotaciones) — insuficientes para aprender
- Undersampling de `normal_skin`, `acne` y `dark_circle` a 6.000 anotaciones cada una
- Remapeo de IDs: wrinkle pasa de ID 8 a ID 6 en el esquema final de 7 clases

Dataset balanceado resultante con ratio **5.5x** (frente al 971x inicial).

**Tres modelos entrenados**

| Modelo | Configuracion | Resultado | Archivo |
|--------|--------------|-----------|---------|
| Modelo 1 | YOLOv8n, patience=5 | Fallo — paro en epoca 21, mAP50=0.0003 | descartado |
| Modelo 2 | YOLOv8m, patience=20, 6k/clase | Mejor resultado | `dermavision_best_v3.pt` |
| Modelo 3 | YOLOv8m, patience=20, 1.1k/clase | Inferior al modelo 2 | `dermavision_best_v4.pt` |

El modelo 1 fallo porque patience=5 detuvo el entrenamiento antes de que el mAP arrancara. Las metricas de mAP en YOLOv8 necesitan mas epocas para superar el umbral de confianza, especialmente con datasets de textura fina como afecciones cutaneas.

El modelo 2 usa **YOLOv8m** (medium) en lugar de nano porque el medium captura mejor los patrones de textura de la piel con sus 25.8M parametros frente a los 3M del nano.

Configuracion del modelo 2:

```python
{
    "model":       "yolov8m.pt",
    "imgsz":       640,
    "batch":       16,
    "epochs":      50,
    "patience":    20,
    "lr0":         0.01,
    "cls":         1.5,
    "mosaic":      1.0,
    "copy_paste":  0.3,
    "seed":        123,
    "device":      0,
    "workers":     2,
}
```

#### Descargar el modelo

Panel derecho de Kaggle → Output → `dermavision_best_v3.pt` → Download.

Copiar a `models/dermavision_best_v3.pt`.

---

## Decisiones tomadas durante el proyecto

**CelebA para normal_skin y dark_circle**: no existe dataset publico de piel normal etiquetada. CelebA con 202k imagenes y atributos binarios verificados fue la unica fuente que permitia filtrar por `Bags_Under_Eyes` para garantizar que las clases no se contaminaran mutuamente.

**No eliminar duplicados aproximados**: el algoritmo de similitud de histograma identifico pares similares pero al visualizarlos las imagenes marcadas para eliminar no eran las correctas. Se decidio conservarlos.

**patience=20 en el modelo final**: el primer entrenamiento con patience=5 paro en epoca 21 con mAP50=0.0003. Las curvas de box_loss y cls_loss mostraban descenso correcto pero el mAP necesitaba mas epocas para arrancar. Con patience=20 el modelo convergio correctamente.

**YOLOv8m en lugar de YOLOv8n**: el nano tenia dificultades con las texturas finas de afecciones cutaneas. El medium ofrece mejor capacidad de representacion sin ser demasiado pesado para inferencia en CPU.

**Eliminar pores y skin_redness**: con 440 y 376 anotaciones respectivamente, YOLOv8 no tiene suficientes ejemplos. Mantenerlas degradaba el rendimiento general sin aportar utilidad real.

**diccionario JSON**: el diccionario estatico para recomendaciones de habitos y productos para la condición detectada

**Groq API Key**: ya que el diccionado Json no podia combinar multiples condiciones simultaneas ni actualizarse con nuevos productos. Groq genera recomendaciones con productos reales y precios adaptados al caso del usuario.

---

## Clases detectadas

| ID | Clase | Descripcion |
|----|-------|-------------|
| 0 | acne | Lesiones inflamatorias del foliculo pilosebaceo |
| 1 | dark circle | Oscurecimiento periorbital (ojeras) |
| 2 | darkspot | Hiperpigmentacion, manchas solares o post-inflamatorias |
| 3 | dry | Piel seca con descamacion o tirantez |
| 4 | normal skin | Piel sin afecciones cutaneas visibles |
| 5 | oily | Exceso de sebo con aspecto brillante |
| 6 | wrinkle | Arrugas y lineas de expresion |

---


## Datasets utilizados

| Dataset | URL | Clases aportadas |
|---------|-----|-----------------|
| Skin Problem Detection v2 | [Roboflow](https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-detection-multiple-clean) | 9 clases base |
| CelebA | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) | normal skin, dark circle |
| face_skin_condition | [Roboflow](https://universe.roboflow.com/muhammad-rifqi-maruf/face_skin_condition) | dry, oily, wrinkle, normal |
| dark_circle | [Roboflow](https://universe.roboflow.com/buyume-ahuro/dark_circle) | dark circle, darkspot, wrinkle |
| darkspot | [Roboflow](https://universe.roboflow.com/skinanalysisv22025/darkspot-jpnki) | darkspot |
| redness | [Roboflow](https://universe.roboflow.com/test-itme1/redness-7qf1p) | skin redness (eliminada en entrenamiento) |

---

###  Limitaciones

* El rendimiento depende de la calidad de la imagen y las condiciones de iluminación.
* El sistema únicamente puede detectar las condiciones incluidas en el conjunto de entrenamiento.
* Las recomendaciones generadas tienen carácter informativo.
* No sustituye la evaluación realizada por un dermatólogo u otro profesional sanitario.
* La precisión del modelo puede variar según las características de los datos utilizados.

---

### Líneas Futuras de Trabajo

* Ampliar el dataset con más condiciones dermatológicas.
* Incorporar imágenes de diferentes tonos y tipos de piel para mejorar la generalización.
* Validar el sistema con profesionales dermatológicos.
* Implementar una aplicación móvil.
* Mejorar el sistema de recomendaciones mediante modelos de IA generativa.
* Incorporar seguimiento temporal de la evolución de la piel.
* Añadir soporte multilingüe.
