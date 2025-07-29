import cv2
import csv
import os
import time # Para calcular FPS
import psutil # Para medir uso de RAM
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
MODEL_PATH = 'runs/detect/train3/weights/best.pt'
VIDEO_PATH = 'video/video.mp4'
LOG_FILE = 'detection_log.csv'
# Dispositivo a usar: 'cuda' para GPU, 'cpu' para CPU

# DEVICE = 'cuda'
DEVICE = 'cpu'

# --- INICIALIZACIÓN ---
# Crea el directorio para las capturas si no existe
os.makedirs('screenshots', exist_ok=True)
process = psutil.Process(os.getpid()) # Proceso actual para medir RAM

try:
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    print(f"Modelo cargado en: {DEVICE}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}"); exit()

# Verifica si el archivo de log ya existe para no repetir la cabecera
log_file_exists = os.path.exists(LOG_FILE)

# Abre el archivo en modo 'a' (anexo) para conservar los registros anteriores
with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not log_file_exists:
        writer.writerow(['timestamp', 'instrument', 'confidence', 'device', 'source_type', 'fps', 'ram_usage_mb'])

# --- FUNCIONES ---
def log_detection(instrument, confidence, device, source_type, fps, ram_mb):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([timestamp, instrument, f"{confidence:.2f}", device, source_type, f"{fps:.2f}", f"{ram_mb:.2f}"])

def list_cameras():
    arr = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened(): arr.append(i)
        cap.release()
    return arr

def run_inference(source):
    delay = 1
    source_type = "camara_web" if isinstance(source, int) else "video_test"

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if isinstance(source, str):
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if fps_video > 0: delay = int(1000 / fps_video)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente de video {source}"); return

    while True:
        start_time = time.time() # Inicia el contador de tiempo
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, device=DEVICE, verbose=False)
        annotated_frame = results[0].plot()

        end_time = time.time() # Finaliza el contador de tiempo
        
        # Calcular FPS y RAM
        fps = 1 / (end_time - start_time)
        ram_usage_mb = process.memory_info().rss / (1024 * 1024) # Convertir a MB

        screenshot_taken_this_frame = False
        for result in results[0].boxes:
            class_id, class_name, confidence = int(result.cls), model.names[int(result.cls)], float(result.conf)
            log_detection(class_name, confidence, DEVICE, source_type, fps, ram_usage_mb)

            if confidence > 0.80 and not screenshot_taken_this_frame:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"screenshots/captura_{class_name}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✅ Captura guardada: {filename}")
                screenshot_taken_this_frame = True
        
        # Mostrar FPS en la ventana
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detector de Instrumentos Ecuatorianos - Presiona 'q' para salir", annotated_frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

# --- MENÚ PRINCIPAL ---
def main_menu():
    while True:
        print("\n--- Detector de Instrumentos Musicales Ecuatorianos ---")
        print("1. Usar cámara web en tiempo real")
        print("2. Analizar video de demostración")
        print("3. Salir")
        choice = input("Selecciona una opción: ")
        if choice == '1':
            cameras = list_cameras()
            if not cameras: print("No se encontraron cámaras web."); continue
            print(f"Cámaras disponibles: {cameras}")
            try:
                cam_choice = int(input("Elige el número de cámara: "))
                if cam_choice in cameras: run_inference(cam_choice)
                else: print("Selección no válida.")
            except ValueError: print("Entrada no válida.")
        elif choice == '2': run_inference(VIDEO_PATH)
        elif choice == '3': print("Saliendo..."); break
        else: print("Opción no válida.")

if __name__ == "__main__":
    main_menu()