#Comandos para instalar as dependências necessárias:
#pip install opencv-python torch ultralytics pyttsx3
#sudo apt install espeak-ng

import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import os

# Inicializa o TTS (text-to-speech)
tts = pyttsx3.init()
tts.setProperty('rate', 160)  # Ajuste da velocidade da fala

# Carrega o modelo YOLOv8 pré-treinado
detector = YOLO("yolov8n.pt")  # YOLOv8 Nano (leve e rápido)

# Abertura de vídeo (usando um arquivo de teste ou webcam)
cap = cv2.VideoCapture('video.mp4')  # Use '0' para webcam ou 'video.mp4' para um arquivo de vídeo

# Configuração da fala
detected_objects = []  # Armazena os objetos que já foram narrados para evitar repetições

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza a detecção de objetos
    results = detector(frame)

    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls)]  # Nome da classe (ex: "pessoa", "carro")
            conf = float(box.conf)  # Confiança da detecção
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do bounding box

            # Desenha o bounding box no vídeo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Se a confiança for maior que 60%, e o objeto ainda não foi narrado, narra o objeto
            if conf > 0.6 and cls not in detected_objects:
                text = f"{cls} detectado à frente"
                os.system(f"espeak-ng '{text}'")  # Usando espeak-ng para narrar
                detected_objects.append(cls)  # Adiciona o objeto à lista de objetos já narrados

    # Exibe o vídeo com os resultados
    cv2.imshow("Detection", frame)

    # Fecha o vídeo quando pressionar a tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:  # 27 é o código da tecla ESC
        break

cap.release()
cv2.destroyAllWindows()
