import os
import socket
import struct
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
 
# ---------------- VARIABLES ----------------
MODEL_PATH = r"C:\Users\QARM4\Desktop\yolo\my_model.pt"
MIN_CONFIDENCE = 0.85
TIMEOUT_THRESHOLD = 0.5  # durée max pour garder last_valid_point si Z=0
GRIP_DELAY =  0.75       # délai avant grip si Z=0
GRIP_HOLD = 1.0           # garder grip actif 3s
 
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
# ---------------- CHARGEMENT YOLO ----------------
if not os.path.exists(MODEL_PATH):
    print(f"ERREUR: Le fichier modèle n'a pas été trouvé : {MODEL_PATH}")
    sys.exit(0)
 
model = YOLO(MODEL_PATH, task='detect')
print("Modèle YOLO chargé.")
 
# ---------------- INITIALISATION REALSENSE ----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
print("[INFO] RealSense pipeline démarré...")
 
# ---------------- VARIABLES ----------------
last_valid_point_3d = [0.0, 0.0, 0.0]
last_detection_time = 0
z_zero_start_time = None
grip_state = 0
grip_trigger_time = None
grip_release_time = None
frame_rate_buffer = []
fps_avg_len = 30
 
# ---------------- BOUCLE PRINCIPALE ----------------
try:
    while True:
        t_start = time.perf_counter()
        current_time = time.time()
 
        point_to_send = [0.0, 0.0, 0.0]
        found_now = False
        distance = None
 
        # Récupération frames
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
 
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        frame = np.asanyarray(color_frame.get_data())
 
        # YOLO detection
        results = model(frame, verbose=False)
        detections = results[0].boxes
 
        for det in detections:
            conf = det.conf.item()
            if conf < MIN_CONFIDENCE:
                continue
 
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            u = int((xmin + xmax) / 2)
            v = int((ymin + ymax) / 2)
            distance = depth_frame.get_distance(u, v)
 
            if distance > 0:
                # Détection normale
                point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], distance)
                last_valid_point_3d = point_3d
                last_detection_time = current_time
                z_zero_start_time = None
                grip_trigger_time = None
                found_now = True
 
            elif distance == 0:
                # Balle trop proche
                if z_zero_start_time is None:
                    z_zero_start_time = current_time
                    grip_trigger_time = current_time
                found_now = True
 
            # Dessin bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
            break  # UNE SEULE BALLE
 
        # ---------------- LOGIQUE GRIP ----------------
        if grip_trigger_time is not None and current_time - grip_trigger_time >= GRIP_DELAY:
            grip_state = 1
            grip_release_time = current_time + GRIP_HOLD  # verrouillage 3s
 
        if grip_release_time is not None and current_time >= grip_release_time:
            grip_state = 0
            grip_trigger_time = None
            grip_release_time = None
 
        # ---------------- LOGIQUE ENVOI X,Y,Z ----------------
        if found_now:
            if distance == 0:
                # Z=0 → envoyer last_valid_point pendant TIMEOUT_THRESHOLD
                if current_time - z_zero_start_time <= TIMEOUT_THRESHOLD:
                    point_to_send = last_valid_point_3d
                else:
                    point_to_send = [0.0, 0.0, 0.0]
            else:
                # Détection normale
                point_to_send = last_valid_point_3d
        else:
            # Pas de détection
            point_to_send = [0.0, 0.0, 0.0]
 
        # ---------------- ENVOI UDP ----------------
        try:
            message_bytes = struct.pack(
                'ffff',
                point_to_send[0],
                point_to_send[1],
                point_to_send[2],
                float(grip_state)
            )
            sock.sendto(message_bytes, (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"Erreur UDP : {e}")
 
        # ---------------- FPS & AFFICHAGE ----------------
        t_stop = time.perf_counter()
        fps = 1.0 / (t_stop - t_start)
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_fps = np.mean(frame_rate_buffer)
 
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"XYZ: {[round(x,2) for x in point_to_send]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"GRIP: {int(grip_state)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("YOLO Ball Detection & Grip", frame)
 
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break
 
finally:
    print("\nFermeture des flux...")
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
    print("Terminé.")
 