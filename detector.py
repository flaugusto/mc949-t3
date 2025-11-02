import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import time
import subprocess
from collections import deque, defaultdict

# ------------ Modelos ------------
detector = YOLO("yolov8n.pt")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ------------ Parâmetros ------------
DETECT_CONF = 0.25
DETECT_IOU = 0.45
DETECT_IMGSZ = 640

# Classes-alvo (COCO → PT-BR)
TARGET_CLASS_MAP = {
    "person": "pessoa",
    "bicycle": "bicicleta",
    "motorbike": "moto",
    "car": "carro",
    "bus": "ônibus",
    "truck": "caminhão",
    "bench": "banco",
    # Pseudo-classe mantida
    "pillar": "pilastra",
}

CLASS_WEIGHTS = {
    "person": 1.0,
    "bicycle": 1.0,
    "motorbike": 1.0,
    "car": 1.0,
    "bus": 1.0,
    "truck": 1.0,
    "bench": 0.4,
    "pillar": 0.9,
}

# Confiabilidades mínimas e restrições por classe
MIN_CONF_PER_CLASS = {
    "person": 0.60,
    "bicycle": 0.35,
    "motorbike": 0.55,
    "car": 0.35,
    "bus": 0.35,
    "truck": 0.35,
    "bench": 0.35,
}
MIN_AREA_FRAC = 0.002  # área mínima relativa do bbox
PERSON_MIN_AR = 1.2     # razão altura/largura mínima para pessoa
MOTORBIKE_MIN_AREA_FRAC = 0.004
MOTORBIKE_MIN_AR = 0.6
MOTORBIKE_MAX_AR = 2.5

# ROI e zona de perigo (frações do frame)
ROI_X = (0.30, 0.70)
ROI_Y = (0.20, 0.90)
DANGER_X = (0.40, 0.60)
DANGER_Y = (0.60, 0.95)

# Bandas de proximidade (0=longe, 1=mais perto)
BAND_VERY_CLOSE = 0.85
BAND_CLOSE = 0.60
BANDS_FOR_LABEL = (0.20, 0.50, 0.80)  # muito perto, perto, médio, longe (via prox invertida)

# Suavização (EMA) para proximidade
EMA_ALPHA = 0.6

# TTS e rate-limiting
GLOBAL_SPEAK_INTERVAL = 1.5
ALERT_COOLDOWN = 1.0
CLASS_COOLDOWN = 2.5

last_speak_time = 0.0
last_alert_time = 0.0
last_class_time = 0.0
last_action_text = ""
last_action_time = 0.0

ema_prox_by_key = {}
label_history_by_loc = defaultdict(lambda: deque(maxlen=5))
last_latency_ms = 0.0

# Fallback por profundidade (parede/pilastra)
PROX_NEAR_THRESHOLD = 0.70
PILLAR_MIN_AR = 2.5
PILLAR_MIN_AREA_FRAC = 0.003
SAME_ACTION_INTERVAL = 2.5
DEPTH_EVERY_N = 2

def speak(text):
    # Non-blocking TTS to avoid stalling the main loop
    try:
        subprocess.Popen(["espeak-ng", "-v", "pt-br", "-s", "170", text])
    except Exception:
        # Fallback (blocking) if Popen fails
        os.system(f"espeak-ng -v pt-br -s 170 '{text}'")

def compute_depth(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        depth_tensor = midas(input_batch)
        depth = depth_tensor.squeeze().cpu().numpy()
    # Suavização leve para reduzir ruído
    depth32 = depth.astype(np.float32)
    depth = cv2.GaussianBlur(depth32, (5, 5), 0)
    # Normalização para visualização
    min_depth = np.percentile(depth, 1)
    max_depth = np.percentile(depth, 99)
    depth_norm = np.clip((depth - min_depth) / (max_depth - min_depth) * 255, 0, 255)
    depth_vis = depth_norm.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    return depth, depth_vis, float(min_depth), float(max_depth)

def detect_objects(frame):
    results = detector(frame, conf=DETECT_CONF, iou=DETECT_IOU, imgsz=DETECT_IMGSZ)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_en = r.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "cls_en": cls_en,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
            })
    return detections

def filter_relevant(detections):
    filtered = []
    for d in detections:
        cls_en = d["cls_en"]
        if cls_en not in TARGET_CLASS_MAP:
            continue
        conf = d["conf"]
        min_conf = MIN_CONF_PER_CLASS.get(cls_en, DETECT_CONF)
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = d["bbox"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if cls_en == "person":
            ar = h / float(w)
            if ar < PERSON_MIN_AR:
                continue
        if cls_en == "motorbike":
            ar = h / float(w)
            if not (MOTORBIKE_MIN_AR <= ar <= MOTORBIKE_MAX_AR):
                continue
        d_out = d.copy()
        d_out["_wh"] = (w, h)
        filtered.append(d_out)
    return filtered

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def in_region(px, py, w, h, rx, ry):
    return (rx[0]*w <= px <= rx[1]*w) and (ry[0]*h <= py <= ry[1]*h)

def estimate_proximity(bbox, depth, min_d, max_d, frame_shape):
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame_shape[:2]
    h_depth, w_depth = depth.shape[:2]
    scale_x = w_depth / w_frame
    scale_y = h_depth / h_frame
    dx1 = int(max(0, x1 * scale_x))
    dy1 = int(max(0, y1 * scale_y))
    dx2 = int(min(w_depth - 1, x2 * scale_x))
    dy2 = int(min(h_depth - 1, y2 * scale_y))
    region = depth[dy1:dy2, dx1:dx2]
    if region.size == 0:
        return 0.0, "longe"
    obj_depth = float(np.median(region))
    # Normaliza (0=longe, 1=perto)
    prox = np.clip((obj_depth - min_d) / (max_d - min_d), 0.0, 1.0)
    prox = 1.0 - prox
    # Rótulo textual
    t1, t2, t3 = BANDS_FOR_LABEL
    if prox >= 1.0 - t1:  # muito perto
        label = "muito perto"
    elif prox >= 1.0 - t2:
        label = "perto"
    elif prox >= 1.0 - t3:
        label = "médio"
    else:
        label = "longe"
    return float(prox), label

def depth_fallback_detections(frame, depth, min_d, max_d):
    h_frame, w_frame = frame.shape[:2]
    h_depth, w_depth = depth.shape[:2]
    # Proximidade em todo o mapa
    prox_full = 1.0 - np.clip((depth - min_d) / (max_d - min_d), 0.0, 1.0)
    # ROI em coordenadas do frame
    fx1 = int(ROI_X[0] * w_frame)
    fx2 = int(ROI_X[1] * w_frame)
    fy1 = int(ROI_Y[0] * h_frame)
    fy2 = int(ROI_Y[1] * h_frame)
    # Mapear ROI para resolução da profundidade
    sx = w_depth / float(w_frame)
    sy = h_depth / float(h_frame)
    dx1 = int(fx1 * sx)
    dx2 = int(fx2 * sx)
    dy1 = int(fy1 * sy)
    dy2 = int(fy2 * sy)
    dx1 = max(0, min(w_depth-1, dx1))
    dx2 = max(0, min(w_depth-1, dx2))
    dy1 = max(0, min(h_depth-1, dy1))
    dy2 = max(0, min(h_depth-1, dy2))
    if dx2 <= dx1 or dy2 <= dy1:
        return []
    roi_prox = prox_full[dy1:dy2, dx1:dx2]
    near_mask = (roi_prox >= PROX_NEAR_THRESHOLD).astype(np.uint8) * 255
    dets = []
    # Pilastras: componentes estreitos e altos
    try:
        contours, _ = cv2.findContours(near_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0 or w <= 0:
                continue
            ar = h / float(w)
            area_frac = (w * h) / float(h_depth * w_depth)
            if ar >= PILLAR_MIN_AR and area_frac >= PILLAR_MIN_AREA_FRAC:
                # Mapear de volta para coordenadas do frame
                fx1_c = int((x + dx1) / sx)
                fy1_c = int((y + dy1) / sy)
                fx2_c = int((x + dx1 + w) / sx)
                fy2_c = int((y + dy1 + h) / sy)
                fx1_c = max(0, min(w_frame-1, fx1_c))
                fx2_c = max(0, min(w_frame-1, fx2_c))
                fy1_c = max(0, min(h_frame-1, fy1_c))
                fy2_c = max(0, min(h_frame-1, fy2_c))
                if fx2_c > fx1_c and fy2_c > fy1_c:
                    dets.append({
                        "cls_en": "pillar",
                        "conf": 0.95,
                        "bbox": (fx1_c, fy1_c, fx2_c, fy2_c),
                    })
    except Exception:
        pass
    return dets

def ema_update(key, value):
    if key not in ema_prox_by_key:
        ema_prox_by_key[key] = value
    else:
        ema_prox_by_key[key] = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * ema_prox_by_key[key]
    return ema_prox_by_key[key]

def class_weight(cls_en):
    return CLASS_WEIGHTS.get(cls_en, 0.5)

def select_primary(dets, frame, depth, min_d, max_d):
    h, w = frame.shape[:2]
    best = None
    best_score = -1.0
    for d in dets:
        bbox = d["bbox"]
        prox, label = estimate_proximity(bbox, depth, min_d, max_d, frame.shape)
        # Suavização por chave (classe + bbox aproximada)
        x1, y1, x2, y2 = bbox
        key = (d["cls_en"], int(x1/10), int(y1/10), int(x2/10), int(y2/10))
        prox_smooth = ema_update(key, prox)
        cx, cy = bbox_center(bbox)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        # filtra por área mínima relativa
        if (area / float(w * h)) < MIN_AREA_FRAC:
            continue
        area_norm = area / float(w * h)
        # área mínima específica por classe
        if d["cls_en"] == "motorbike" and area_norm < MOTORBIKE_MIN_AREA_FRAC:
            continue
        roi_bonus = 1.0 if in_region(cx, cy, w, h, ROI_X, ROI_Y) else 0.0
        score = 0.5 * prox_smooth + 0.25 * area_norm + 0.15 * roi_bonus + 0.10 * class_weight(d["cls_en"])
        if score > best_score:
            d_out = d.copy()
            d_out["prox"] = prox_smooth
            d_out["prox_label"] = label
            d_out["center"] = (cx, cy)
            d_out["score"] = score
            best_score = score
            best = d_out
    return best

def assess_collision(target, frame):
    if target is None:
        return None, "low"
    h, w = frame.shape[:2]
    cx, cy = target["center"]
    prox = target["prox"]
    danger = in_region(cx, cy, w, h, DANGER_X, DANGER_Y)
    if danger and prox >= BAND_VERY_CLOSE:
        return "Pare", "high"
    if prox >= BAND_CLOSE:
        # Desvie para o lado oposto do alvo
        side = "direita" if cx < 0.5 * w else "esquerda"
        return f"Desvie à {side}", "med"
    return None, "low"

def update_and_get_smoothed_class(target):
    # Agrupa por localização aproximada do bbox (independente da classe)
    x1, y1, x2, y2 = target["bbox"]
    loc_key = (int(x1/20), int(y1/20), int(x2/20), int(y2/20))
    pt_label = TARGET_CLASS_MAP.get(target["cls_en"], target["cls_en"])
    hist = label_history_by_loc[loc_key]
    hist.append(pt_label)
    # Maioria simples
    counts = {}
    for v in hist:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

def can_speak(now, kind, action_text=None):
    global last_speak_time, last_alert_time, last_class_time, last_action_text, last_action_time
    if now - last_speak_time < GLOBAL_SPEAK_INTERVAL:
        return False
    if kind == "alert":
        # Evita repetir o mesmo alerta em sequência
        if action_text and action_text == last_action_text and (now - last_action_time) < SAME_ACTION_INTERVAL:
            return False
        return now - last_alert_time >= ALERT_COOLDOWN
    if kind == "class":
        return now - last_class_time >= CLASS_COOLDOWN and (now - last_alert_time) >= 0.5
    return False

def mark_spoken(now, kind, action_text=None):
    global last_speak_time, last_alert_time, last_class_time, last_action_text, last_action_time
    last_speak_time = now
    if kind == "alert":
        last_alert_time = now
        if action_text:
            last_action_text = action_text
            last_action_time = now
    if kind == "class":
        last_class_time = now

def draw_overlay(frame, depth_vis, target, fps, action, latency_ms):
    vis = frame
    # Desenhar alvo e info
    if target is not None:
        x1, y1, x2, y2 = target["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cls_pt = target.get("cls_pt", TARGET_CLASS_MAP.get(target["cls_en"], target["cls_en"]))
        txt = f"{cls_pt} | {target['prox_label']} | {target['prox']:.2f}"
        cv2.putText(vis, txt, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    # FPS e ação
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    if action:
        cv2.putText(vis, action, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    # Latência até a fala mais recente (ms)
    cv2.putText(vis, f"LAT: {latency_ms:.0f} ms", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    # Mostrar
    cv2.imshow("Detection", vis)
    cv2.imshow("Depth", depth_vis)

# ------------ Vídeo ------------
cap = cv2.VideoCapture('video.mp4')

prev_time = time.perf_counter()
frame_idx = 0
cached_depth = None
cached_depth_vis = None
cached_min_d = None
cached_max_d = None

while cap.isOpened():
    loop_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    # Depth decimation
    if frame_idx % DEPTH_EVERY_N == 0 or cached_depth is None:
        depth, depth_vis, min_d, max_d = compute_depth(frame)
        cached_depth, cached_depth_vis = depth, depth_vis
        cached_min_d, cached_max_d = min_d, max_d
    else:
        depth, depth_vis, min_d, max_d = cached_depth, cached_depth_vis, cached_min_d, cached_max_d
    all_dets = detect_objects(frame)
    relevant = filter_relevant(all_dets)
    # Fallback por profundidade para paredes/pilastras
    pseudo = depth_fallback_detections(frame, depth, min_d, max_d)
    if pseudo:
        relevant.extend(pseudo)
    primary = select_primary(relevant, frame, depth, min_d, max_d)
    if primary is not None:
        # Suavizar classe textual
        primary["cls_pt"] = update_and_get_smoothed_class(primary)
    action, priority = assess_collision(primary, frame)

    now = time.perf_counter()
    if action and can_speak(now, "alert", action_text=action):
        latency_ms = (now - loop_start) * 1000.0
        speak(action)
        mark_spoken(now, "alert", action_text=action)
        last_latency_ms = latency_ms
    elif primary is not None and can_speak(now, "class") and action is None:
        latency_ms = (now - loop_start) * 1000.0
        cls_pt = primary.get("cls_pt", TARGET_CLASS_MAP.get(primary["cls_en"], primary["cls_en"]))
        speak(f"{cls_pt}, {primary['prox_label']}")
        mark_spoken(now, "class")
        last_latency_ms = latency_ms

    # FPS
    dt = now - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = now

    draw_overlay(frame, depth_vis, primary, fps, action, last_latency_ms)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
