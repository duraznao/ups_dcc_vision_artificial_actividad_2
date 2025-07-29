Claro, aquí tienes una propuesta completa para el archivo `README.md`. Este documento está diseñado para ser claro y guiar a cualquier persona que encuentre tu proyecto en GitHub, basándose en nuestra exitosa implementación en Windows.

-----

# Detector de Instrumentos Musicales Andinos con YOLOv10

Este proyecto utiliza el modelo de detección de objetos **YOLOv10** para identificar 5 instrumentos musicales andinos a través de *transfer learning*. La aplicación es capaz de realizar detecciones en tiempo real mediante una cámara web o analizar un archivo de video preexistente.

Los instrumentos que el modelo puede reconocer son:

  * Bombo Andino (etiqueta: bombo_andino)
  * Charango (etiqueta: charango)
  * Dulzaina (etiqueta: dulzaina)
  * Pingullo (etiqueta: pingullo)
  * Rondador (etiqueta: rondador)

 ## Características

  * **Detección en tiempo real:** Utiliza la cámara web para identificar los instrumentos en vivo.
  * **Análisis de video:** Procesa un archivo de video para detectar los objetos.
  * **Registro de rendimiento:** Guarda cada detección en un archivo `detection_log.csv` con información de fecha, hora, objeto, confianza y si se usó CPU o GPU, para análisis posteriores.
  * **Optimizado para windows:** Todo el flujo de trabajo ha sido validado y ajustado para funcionar en un entorno Windows con aceleración por GPU.

-----

## Requisitos

### Hardware

  * Una tarjeta gráfica **NVIDIA compatible con CUDA** es esencial para el entrenamiento y la inferencia acelerada.

### Software

  * **Sistema Operativo:** Windows 10 / 11.
  * **Python:** Versión **3.11 (64-bit)**. Versiones más recientes como 3.12+ pueden tener problemas de compatibilidad con PyTorch.
  * **OpenCV:** 
-----

## Instalación y puesta en marcha

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

### 2. Estructura de directorios (¡importante!)

Este repositorio no incluye los datos de entrenamiento. Debe crear dos directorios en la raíz del proyecto y organizarlos de la siguiente manera:

  * **`dataset/`**: Contiene las imágenes `.jpg`.
  * **`etiquetas/`**: Contiene los archivos `.txt` exportados desde CVAT en formato YOLO.

<!-- end list -->

```
proyecto_yolo_ecuador/
├── dataset/
│   ├── bombo_andino/
│   │   └── bombo_andino_001.jpg
│   │   └── ...
│   ├── charango/
│   └── ...
├── etiquetas/
│   ├── bombo_andino/
│   │   └── obj_train_data/
│   │       └── bombo_andino_001.txt
│   │       └── ...
│   ├── charango/
│   └── ...
└── videos/
    └── demo.mp4  (coloca aquí tu video de prueba)
```

### 3. Configurar el entorno virtual

```cmd
python -m venv env
.\env\Scripts\activate
```

### 4. Instalar dependencias

Este paso instala PyTorch con soporte para CUDA 12.1 de forma automática.

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python
```

### 5. Preparar el _dataset_

Ejecutar el script para organizar las imágenes y las etiquetas en el formato que YOLO necesita para entrenar.

```cmd
python preparar_dataset.py
```

Esto creará una carpeta `instrumentos_dataset/` y un archivo `instrumentos.yaml`.

### 6. Entrenar el modelo

Ahora, se realiza el fine-tuning del modelo. Este proceso puede tardar dependiendo de la GPU.

```cmd
python train.py
```

El modelo entrenado se guardará en una carpeta similar a `runs/detect/train/weights/best.pt`.

-----

## Aplicación

1.  Abrir el archivo `main.py` y actualizar la variable `MODEL_PATH` con la ruta al modelo entrenado (ej. `'runs/detect/train/weights/best.pt'`).
2.  Ejecutar la aplicación desde la terminal:
    ```cmd
    python main.py
    ```
3.  Usar el menú de la aplicación para elegir entre la cámara web o el análisis de video. Para salir de la visualización, presionar la tecla `q`.

-----

## Licencia

Este proyecto se distribuye bajo la Licencia MIT.