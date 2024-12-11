import numpy as np
import cv2
import tensorflow.lite as tflite
import time
import psutil
import subprocess
import threading
import pandas as pd
import os
from datetime import datetime

# Ruta al modelo y al video
MODEL_PATH = r"/home/joshua/Desktop/ruta_al_modelo/model.tflite"
VIDEO_PATH = r"/home/joshua/Desktop/ruta_al_modelo/video.mp4"
LOG_FILE = "metrics_log.csv"
REPORTE_FILE = "metrics_report.csv"

# Intentar importar el GPU Delegate
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    # Proporcionar la ruta completa a la biblioteca GPU Delegate
    GPU_DELEGATE_PATH = '/ruta/completa/a/libtensorflowlite_gpu_delegate.so'  # Actualiza esta ruta
    if not os.path.exists(GPU_DELEGATE_PATH):
        raise FileNotFoundError(f"GPU Delegate no encontrado en {GPU_DELEGATE_PATH}")
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate(GPU_DELEGATE_PATH)]
    )
except (ImportError, FileNotFoundError) as e:
    print(f"Error al cargar el GPU Delegate: {e}")
    print("Usando el intérprete de TensorFlow Lite sin GPU.")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Etiquetas (emociones)
etiquetas = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Tamaño esperado de las imágenes (128x128)
img_size = (128, 128)

# Función para preprocesar imágenes capturadas del video
def preprocesar_imagen(imagen):
    imagen = cv2.resize(imagen, img_size)  # Redimensionar a 128x128
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    imagen = imagen / 255.0  # Normalización
    imagen = np.expand_dims(imagen, axis=0)  # Agregar dimensión para batch
    imagen = imagen.astype(np.float32)  # Asegurarse de que esté en float32
    return imagen

# Función para monitorear tegrastats y guardar en archivo
def monitorear_tegrastats(log_queue, interval=1):
    try:
        # Ejecutar tegrastats y capturar la salida
        proceso = subprocess.Popen(['tegrastats', '--interval', str(interval * 1000)], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        while True:
            line = proceso.stdout.readline()
            if line:
                log_queue.append(line.strip())
            else:
                break
    except Exception as e:
        log_queue.append(f"Error: {e}")

# Función para estimar FLOPS del modelo
def estimar_flops(interpreter):
    # Esta es una estimación simplificada.
    try:
        profile = interpreter.get_tensor_details()
        total_flops = 0
        for tensor in profile:
            shape = tensor['shape']
            if 'depthwise' in tensor['name'].lower():
                total_flops += np.prod(shape) * 2
            else:
                total_flops += np.prod(shape) * 2
        return total_flops
    except:
        return "No disponible"

# Función para registrar métricas en el archivo
def registrar_metricas(timestamp, fps, cpu, mem_used, mem_total, flops, tegrastats):
    df = pd.DataFrame([{
        'Timestamp': timestamp,
        'FPS': fps,
        'CPU_Usage(%)': cpu,
        'Memory_Used_GB': mem_used,
        'Memory_Total_GB': mem_total,
        'FLOPS': flops,
        'Tegrastats': tegrastats
    }])
    df.to_csv(LOG_FILE, mode='a', header=False, index=False)

# Crear/Inicializar el archivo de log
if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=[
        'Timestamp', 'FPS', 'CPU_Usage(%)', 'Memory_Used_GB', 
        'Memory_Total_GB', 'FLOPS', 'Tegrastats'
    ])
    df.to_csv(LOG_FILE, index=False)

# Crear una lista para almacenar la salida de tegrastats
tegrastats_log = []

# Iniciar el hilo de monitoreo de tegrastats
hilo_tegrastats = threading.Thread(target=monitorear_tegrastats, args=(tegrastats_log,), daemon=True)
hilo_tegrastats.start()

# Configuración del video
cap = cv2.VideoCapture(r"/home/joshua/Desktop/ruta_al_modelo/video.mp4")

# Comprobar si se pudo abrir el video
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Inicializar variables para calcular FPS
fps_start_time = time.time()
fps = 0
total_frames = 0

# Crear una ventana llamada 'Detección de Emociones en Tiempo Real'
cv2.namedWindow('Detección de Emociones en Tiempo Real', cv2.WINDOW_NORMAL)

while True:
    # Capturar frame a frame
    ret, frame = cap.read()

    if not ret:
        print("Fin del video o error al capturar el frame.")
        break

    # Preprocesar el frame capturado
    input_data = preprocesar_imagen(frame)

    # Configurar los datos de entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Medir tiempo de inicio para calcular FPS
    start_time = time.time()

    # Realizar la inferencia
    interpreter.invoke()

    # Calcular el tiempo de inferencia
    end_time = time.time()
    inferencia_time = end_time - start_time
    fps = 1 / inferencia_time

    # Obtener resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediccion = np.argmax(output_data)  # Obtener la etiqueta predicha
    etiqueta_predicha = etiquetas[prediccion]

    # Obtener métricas del sistema
    cpu_usage = psutil.cpu_percent(interval=None)
    memoria = psutil.virtual_memory()
    memoria_total = memoria.total / (1024 ** 3)  # GB
    memoria_usada = memoria.used / (1024 ** 3)  # GB
 # Obtener la última entrada de tegrastats
    tegrastats_info = tegrastats_log[-1] if tegrastats_log else "No disponible"

    # Estimar FLOPS
    flops = estimar_flops(interpreter)
# Registrar las métricas
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registrar_metricas(timestamp, fps, cpu_usage, memoria_usada, memoria_total, flops, tegrastats_info)

    # Mostrar la etiqueta en la imagen capturada junto con los FPS y métricas
    cv2.putText(frame, f'Predicción: {etiqueta_predicha}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'CPU Uso: {cpu_usage}%', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f'Memoria: {memoria_usada:.2f}GB / {memoria_total:.2f}GB', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f'FLOPS: {flops}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    #cv2.putText(frame, f'Tegrastats: {tegrastats_info}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Mostrar el frame en la ventana predefinida
    cv2.imshow('Detección de Emociones en Tiempo Real', frame)

    # Salir si presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

# Generar el reporte de métricas
print("Generando reporte de métricas...")

# Leer el archivo de log
df = pd.read_csv(LOG_FILE)

# Calcular promedios
promedios = df.mean(numeric_only=True)

# Crear un DataFrame para los promedios
promedios_df = pd.DataFrame([promedios])
promedios_df['Timestamp'] = 'Promedios'
promedios_df['Tegrastats'] = 'Promedios'

# Concatenar los promedios al final del DataFrame original
df_final = pd.concat([df, promedios_df], ignore_index=True)

# Guardar el reporte
df_final.to_csv(REPORTE_FILE, index=False)

print(f"Reporte generado: {REPORTE_FILE}")