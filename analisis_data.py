import pandas as pd

def analizar_por_dispositivo(filepath='detection_log.csv'):
    """
    Analiza el log de detecciones y genera tablas de resultados separadas
    para CPU y GPU, guardándolas en archivos CSV.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filepath}'.")
        return

    # Calcular las estadísticas generales agrupando por ambos campos
    stats_generales = df.groupby(['device', 'instrument'])['confidence'].agg(
        Detecciones_Totales='count',
        Confianza_Minima='min',
        Confianza_Maxima='max',
        Confianza_Promedio='mean'
    ).reset_index()

    # --- Filtrar y Guardar Resultados por Dispositivo ---

    # 1. Tabla y CSV para CPU
    cpu_stats = stats_generales[stats_generales['device'] == 'cpu']
    if not cpu_stats.empty:
        cpu_filename = 'resultados_analisis_cpu.csv'
        cpu_stats.to_csv(cpu_filename, index=False)
        print("\n--- ✅ Resultados para CPU ---")
        print(f"(Guardados en {cpu_filename})")
        print(cpu_stats.to_string())
    else:
        print("\nNo se encontraron registros para CPU.")

    # 2. Tabla y CSV para GPU (CUDA)
    gpu_stats = stats_generales[stats_generales['device'] == 'cuda']
    if not gpu_stats.empty:
        gpu_filename = 'resultados_analisis_gpu.csv'
        gpu_stats.to_csv(gpu_filename, index=False)
        print("\n--- ✅ Resultados para GPU (CUDA) ---")
        print(f"(Guardados en {gpu_filename})")
        print(gpu_stats.to_string())
    else:
        print("\nNo se encontraron registros para GPU (CUDA).")

    # --- 3. Análisis de Rendimiento (FPS y RAM) ---
    print("\n--- 3. Análisis de Rendimiento (FPS y RAM) ---")

    # Agrupar por dispositivo y calcular el promedio de FPS y RAM
    performance_stats = df.groupby('device')[['fps', 'ram_usage_mb']].mean().reset_index()
    performance_stats.rename(columns={'fps': 'FPS Promedio', 'ram_usage_mb': 'Uso RAM Promedio (MB)'}, inplace=True)

    # Guardar los resultados de rendimiento en un nuevo archivo CSV
    output_filename_perf = 'analisis_rendimiento_fps_ram.csv'
    performance_stats.to_csv(output_filename_perf, index=False)
    print(f"Resultados de rendimiento guardados en: {output_filename_perf}")

    # Imprimir la tabla en la consola
    print(performance_stats.to_string())

if __name__ == '__main__':
    analizar_por_dispositivo()