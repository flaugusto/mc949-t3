import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

# ------------ YOLO ------------
detector = YOLO("yolov8n.pt")

# ------------ MiDaS ------------
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ------------ Video ------------
cap = cv2.VideoCapture('video.mp4')

# Objetos já narrados para evitar spam
narrated = set()

def speak(text):
    os.system(f"espeak-ng '{text}'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ------ Depth estimation ------
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        depth_tensor = midas(input_batch)
        depth = depth_tensor.squeeze().cpu().numpy()

    # Normalize depth map for visualization
    min_depth, max_depth = np.percentile(depth, 1), np.percentile(depth, 99)
    depth_norm = np.clip((depth - min_depth) / (max_depth - min_depth) * 255, 0, 255)
    depth_vis = depth_norm.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # ------ YOLO detection ------
    results = detector(frame, conf=0.25, iou=0.45, imgsz=640)

    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h_frame, w_frame = frame.shape[:2]
            h_depth, w_depth = depth.shape[:2]

            scale_x = w_depth / w_frame
            scale_y = h_depth / h_frame

            
            # Scale to depth resolution
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)

            # Depth inside bounding box
            region = depth[dy1:dy2, dx1:dx2]
            if region.size > 0:
                obj_depth = np.median(region)
            else:
                obj_depth = 0

            # Normalize object depth (0 = closest, 1 = farthest)
            obj_depth = np.clip((obj_depth - min_depth) / (max_depth - min_depth), 0, 1)
            obj_depth = 1.0 - obj_depth  # 1 = closest

            # Classifica
            if obj_depth < 0.2:
                distance_label = "muito perto"
            elif obj_depth < 0.5:
                distance_label = "perto"
            elif obj_depth < 0.8:
                distance_label = "médio"
            else:
                distance_label = "longe"


            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls} {obj_depth:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Narrate once per object type
            if conf > 0.6 and cls not in narrated:
                speak(f"{cls} {distance_label}")
                narrated.add(cls)

    # Show video and depth map
    cv2.imshow("Detection", frame)
    cv2.imshow("Depth", depth_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
