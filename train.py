from ultralytics import YOLO

# Este bloque asegura que el código solo se ejecute cuando corres el script directamente.
if __name__ == '__main__':
    # Carga un modelo YOLOv10 pre-entrenado
    model = YOLO('yolov10n.pt') 

    # Entrena el modelo usando nuestro dataset
    print("Iniciando el entrenamiento del modelo...")
    results = model.train(
        data='instrumentos.yaml',
        epochs=75,
        batch=8,
        imgsz=640,
        device=0  # '0' para usar la GPU
    )
    print("¡Entrenamiento completado!")