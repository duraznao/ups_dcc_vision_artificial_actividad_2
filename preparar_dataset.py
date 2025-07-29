import os
import shutil
import random

# --- Configuración Inicial ---
print("Iniciando la preparación del dataset...")
class_names = ["bombo_andino", "charango", "dulzaina", "pingullo", "rondador"]
base_dir = os.getcwd()
image_dir = os.path.join(base_dir, "dataset")
label_base_dir = os.path.join(base_dir, "etiquetas")
output_dir = os.path.join(base_dir, "instrumentos_dataset")
split_ratio = 0.8

# --- Creación de Directorios ---
print(f"Creando la estructura de directorios en: {output_dir}")
for split in ['train', 'valid']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# --- División y Copia de Archivos ---
print("Dividiendo los datos...")
for class_name in class_names:
    image_class_dir = os.path.join(image_dir, class_name)
    label_class_dir = os.path.join(label_base_dir, class_name)
    all_files = [f for f in os.listdir(image_class_dir) if f.endswith('.jpg')]
    random.shuffle(all_files)
    split_index = int(len(all_files) * split_ratio)
    train_files, valid_files = all_files[:split_index], all_files[split_index:]

    def copy_files(files, split_type):
        for filename in files:
            basename = os.path.splitext(filename)[0]
            src_image = os.path.join(image_class_dir, f"{basename}.jpg")
            src_label = os.path.join(label_class_dir, 'obj_train_data', f"{basename}.txt")
            if os.path.exists(src_label):
                shutil.copy(src_image, os.path.join(output_dir, split_type, 'images', f"{basename}.jpg"))
                shutil.copy(src_label, os.path.join(output_dir, split_type, 'labels', f"{basename}.txt"))

    copy_files(train_files, 'train')
    copy_files(valid_files, 'valid')

# --- Creación del Archivo YAML ---
print("Creando el archivo de configuración 'instrumentos.yaml'...")
yaml_path = os.path.join(base_dir, 'instrumentos.yaml')
dataset_path = os.path.abspath(output_dir) # Usar ruta absoluta para robustez

yaml_content = f"""
path: {dataset_path}
train: train/images
val: valid/images

names:
  - bombo_andino
  - charango
  - dulzaina
  - pingullo
  - rondador
"""
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("\n¡Proceso de preparación finalizado con éxito! ✅")