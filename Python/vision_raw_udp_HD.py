import os
import socket
import struct
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

import mediapipe as mp
from hands_detector import (
    es_apuntar_derecha, es_apuntar_izquierda, es_ok, es_palm_stop
)

# Setup global parameters 
MODEL_PATH = r"C:\Users\piopi\Desktop\Centrale\PP S8 Robotique\codes\catching the ball\my_model.pt"
MIN_CONFIDENCE = 0.85
BALL_DIAMETER = 0.067
BALL_RADIUS = BALL_DIAMETER / 2
DEPTH_KERNEL_SIZE = 5
half_k = DEPTH_KERNEL_SIZE // 2

# UDP configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Loading YOLO model
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: El modelo no se encuentra: {MODEL_PATH}")
    sys.exit(0)
model = YOLO(MODEL_PATH, task='detect')
print("YOLO model loaded.")

# ===================== GESTURE: INITIALIZE MEDIAPIPE =====================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

gesture_history = []
HIST_SIZE = 5
# =========================================================================

# RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
print("[INFO] RealSense pipeline started...")


GESTO_TO_INT = {
    "NONE": 0,
    "STOP": 1,
    "OK": 2,
    "RIGHT": 3,
    "LEFT": 4,
    "HAND": 5   # optional
}

def obtener_gesto(hand_landmarks):
    """Devuelve el string del gesto detectado según prioridad."""
    if es_palm_stop(hand_landmarks):
        return "STOP"
    elif es_apuntar_derecha(hand_landmarks):
        return "RIGHT"
    elif es_apuntar_izquierda(hand_landmarks):
        return "LEFT"
    elif es_ok(hand_landmarks):
        return "OK"
    else:
        return "HAND"   

# ===================== PRINCIPAL LOOP =====================
try:
    while True:
        t_start = time.perf_counter()
        
        # Reset variables
        point_sensor = [0.0, 0.0, 0.0]
        point_pinhole = [0.0, 0.0, 0.0]
        found_now = False

        # 1. Capturar frames
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        frame = np.asanyarray(color_frame.get_data())   # imagen BGR

        # ===================== GESTOS: DETECTAR MANOS =====================
        gesto_str = "NONE"
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands_detector.process(rgb_frame)
        
        if results_hands.multi_hand_landmarks:
            # Tomamos la primera mano detectada
            hand_landmarks = results_hands.multi_hand_landmarks[0]
            # Dibujar landmarks (opcional, consume un poco de CPU)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            # Obtener gesto actual
            gesto_actual = obtener_gesto(hand_landmarks)
            
            # Suavizado temporal
            gesture_history.append(gesto_actual)
            if len(gesture_history) > HIST_SIZE:
                gesture_history.pop(0)
            # Gestor final = el más frecuente en el historial
            from collections import Counter
            gesto_str = Counter(gesture_history).most_common(1)[0][0]
        else:
            # No hay mano → reiniciar historial
            gesture_history.clear()
            gesto_str = "NONE"
        
        # Mostrar el gesto en la imagen
        cv2.putText(frame, f"Gesto: {gesto_str}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # =================================================================

        # 2. YOLO Detection (sin cambios)
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # 3. Procesar detecciones (igual que antes)
        if detections is not None:
            for det in detections:
                conf = det.conf.item()
                if conf < MIN_CONFIDENCE:
                    continue

                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                u = int((xmin + xmax) / 2)
                v = int((ymin + ymax) / 2)
                w_pixels = xmax - xmin

                # Profundidad por sensor
                depths = []
                for i in range(-half_k, half_k + 1):
                    for j in range(-half_k, half_k + 1):
                        if 0 <= u + i < 640 and 0 <= v + j < 480:
                            d = depth_frame.get_distance(u + i, v + j)
                            if d > 0:
                                depths.append(d)
                sensor_distance = np.median(depths) if depths else 0.0

                # Distancia pinhole
                pinhole_distance = 0.0
                if w_pixels > 0:
                    fx = intr.fx
                    pinhole_distance = (fx * BALL_DIAMETER) / w_pixels

                # Puntos 3D
                if sensor_distance > 0:
                    point_sensor = rs.rs2_deproject_pixel_to_point(intr, [u, v], sensor_distance + BALL_RADIUS)
                if pinhole_distance > 0:
                    point_pinhole = rs.rs2_deproject_pixel_to_point(intr, [u, v], pinhole_distance)

                if sensor_distance > 0 or pinhole_distance > 0:
                    found_now = True

                # Dibujado (sin cambios)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                dist_str = f"S:{sensor_distance:.2f}m | P:{pinhole_distance:.2f}m"
                (text_width, text_height), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_x = xmax + 5
                if text_x + text_width > 640:
                    text_x = xmin - 5 - text_width
                    if text_x < 0:
                        text_x = 5
                cv2.putText(frame, dist_str, (text_x, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                break  # solo la primera bola

        # 4. UDP Communication 
        try:
            found_now_flag = 1.0 if found_now else 0.0
            gesto_float = float(GESTO_TO_INT.get(gesto_str, 0))
            message_bytes = struct.pack(
                'ffffffff',   # 8 floats: Sensor XYZ, Pinhole XYZ, Detection Flag, Gesto
                float(point_sensor[0]),  # 1
                float(point_sensor[1]),  # 2
                float(point_sensor[2]),  # 3
                float(point_pinhole[0]), # 4
                float(point_pinhole[1]), # 5
                float(point_pinhole[2]), # 6
                found_now_flag,          # 7
                gesto_float              # 8 ← 0=NONE,1=STOP,2=OK,3=RIGHT,4=LEFT
            )
            sock.sendto(message_bytes, (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"UDP Error: {e}")

        # 5. Mostrar FPS y estado
        fps = 1.0 / (time.perf_counter() - t_start)
        cv2.putText(frame, f"Raw FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_txt = "SENDING DUAL RAW DATA" if found_now else "TARGET LOST (SENDING 0s)"
        status_col = (0, 255, 0) if found_now else (0, 0, 255)
        cv2.putText(frame, status_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

        cv2.imshow("YOLO Ball Detection + Gestos", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

finally:
    print("\nClosing...")
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
    print("Done.")