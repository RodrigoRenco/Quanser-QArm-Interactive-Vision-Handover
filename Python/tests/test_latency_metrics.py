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
from collections import Counter
import matplotlib.pyplot as plt 

# Set up the path for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from hands_detector import (
    es_apuntar_derecha,
    es_apuntar_izquierda,
    es_palm_stop
)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# CONFIG
MODEL_PATH = r"Detection Models\model_bottle.pt"
MIN_CONFIDENCE = 0.85
BALL_DIAMETER = 0.067
BALL_RADIUS = BALL_DIAMETER / 2
DEPTH_KERNEL_SIZE = 5
half_k = DEPTH_KERNEL_SIZE // 2

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH, task='detect')

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

gesture_history = []
HIST_SIZE = 5

GESTO_TO_INT = {
    "NONE": 0, "STOP": 1, "HAND": 2, "RIGHT": 3, "LEFT": 4,
}

def obtener_gesto(hand_landmarks, tipo_mano):
    if es_palm_stop(hand_landmarks): return "STOP"
    elif es_apuntar_derecha(hand_landmarks, tipo_mano): return "RIGHT"
    elif es_apuntar_izquierda(hand_landmarks, tipo_mano): return "LEFT"
    else: return "HAND"

# REALSENSE & PROFILING SETUP
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

print("[INFO] RealSense pipeline started...")
print("[INSTRUCTION] Move bottle and do gestures for 15s, then press 'Q' to generate metrics.")

# Arrays to hold our time deltas
prof_realsense = []
prof_hands = []
prof_yolo = []
prof_overhead = []

# MAIN LOOP
try:
    while True:
        t0 = time.perf_counter() # <--- START OF CYCLE

        # 1. REALSENSE HARDWARE FETCH
        point_sensor = [0.0, 0.0, 0.0]
        point_pinhole = [0.0, 0.0, 0.0]
        found_now = False

        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        frame = np.asanyarray(color_frame.get_data())
        
        t1 = time.perf_counter() # <--- END OF REALSENSE

        # 2. MEDIAPIPE GESTURE RECOGNITION
        gesto_str = "NONE"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands_detector.process(rgb_frame)

        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            hand_landmarks = results_hands.multi_hand_landmarks[0]
            handedness = results_hands.multi_handedness[0]
            tipo_mano = handedness.classification[0].label

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesto_actual = obtener_gesto(hand_landmarks, tipo_mano)
            gesture_history.append(gesto_actual)

            if len(gesture_history) > HIST_SIZE:
                gesture_history.pop(0)

            gesto_str = Counter(gesture_history).most_common(1)[0][0]
        else:
            gesture_history.clear()
            gesto_str = "NONE"

        cv2.putText(frame, f"Gesto: {gesto_str}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        t2 = time.perf_counter() # <--- END OF MEDIAPIPE

        # 3. YOLO OBJECT DETECTION & DEPTH EXTRACTION
        results = model(frame, verbose=False)
        detections = results[0].boxes

        if detections is not None:
            for det in detections:
                conf = det.conf.item()
                if conf < MIN_CONFIDENCE: continue

                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                u, v = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
                w_pixels = xmax - xmin

                depths = []
                for i in range(-half_k, half_k + 1):
                    for j in range(-half_k, half_k + 1):
                        if 0 <= u + i < 640 and 0 <= v + j < 480:
                            d = depth_frame.get_distance(u + i, v + j)
                            if d > 0: depths.append(d)

                sensor_distance = np.median(depths) if depths else 0.0
                pinhole_distance = 0.0
                
                if w_pixels > 0:
                    fx = intr.fx
                    pinhole_distance = (fx * BALL_DIAMETER) / w_pixels

                if sensor_distance > 0:
                    point_sensor = rs.rs2_deproject_pixel_to_point(intr, [u, v], sensor_distance + BALL_RADIUS)
                if pinhole_distance > 0:
                    point_pinhole = rs.rs2_deproject_pixel_to_point(intr, [u, v], pinhole_distance)

                if sensor_distance > 0 or pinhole_distance > 0:
                    found_now = True

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                break
                
        t3 = time.perf_counter() # <--- END OF YOLO

        # 4. UDP OVERHEAD & DRAWING
        try:
            found_now_flag = 1.0 if found_now else 0.0
            gesto_float = float(GESTO_TO_INT.get(gesto_str, 0))

            message_bytes = struct.pack(
                'ffffffff',
                float(point_sensor[0]), float(point_sensor[1]), float(point_sensor[2]),
                float(point_pinhole[0]), float(point_pinhole[1]), float(point_pinhole[2]),
                found_now_flag, gesto_float
            )
            sock.sendto(message_bytes, (UDP_IP, UDP_PORT))
        except Exception as e:
            pass # Ignore UDP drops during profiling

        fps = 1.0 / (time.perf_counter() - t0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("YOLO Ball Detection + Gestos", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break
            
        t4 = time.perf_counter() # <--- END OF OVERHEAD

        # Save the time deltas for this frame
        prof_realsense.append(t1 - t0)
        prof_hands.append(t2 - t1)
        prof_yolo.append(t3 - t2)
        prof_overhead.append(t4 - t3)

finally:
    print("\nClosing Camera...")
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
    
    # GENERATE METRICS REPORT
    if len(prof_yolo) > 30: # Ensure we collected enough frames
        avg_rs = np.mean(prof_realsense) * 1000      # Convert to ms
        avg_hands = np.mean(prof_hands) * 1000
        avg_yolo = np.mean(prof_yolo) * 1000
        avg_over = np.mean(prof_overhead) * 1000
        
        total_latency = avg_rs + avg_hands + avg_yolo + avg_over
        true_fps = 1000.0 / total_latency
        
        print("\n" + "="*40)
        print("     PERCEPTION LATENCY METRICS")
        print("="*40)
        print(f"Total Frames Profiled: {len(prof_yolo)}")
        print(f"RealSense Fetch:       {avg_rs:.1f} ms")
        print(f"MediaPipe Hands:       {avg_hands:.1f} ms")
        print(f"YOLOv8 Detection:      {avg_yolo:.1f} ms")
        print(f"Math & UDP Overhead:   {avg_over:.1f} ms")
        print("-" * 40)
        print(f"Total Pipeline Latency: {total_latency:.1f} ms")
        print(f"True Average FPS:       {true_fps:.1f} FPS")
        print("="*40)

        # Generate the Pie Chart
        labels = ['Camera Fetch', 'MediaPipe\n(Gestures)', 'YOLOv8\n(Bottle)', 'UDP & Draw']
        sizes = [avg_rs, avg_hands, avg_yolo, avg_over]
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
        explode = (0, 0, 0.1, 0)  # slightly "explode" the YOLO slice 

        plt.figure(figsize=(9, 6))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140, textprops={'fontsize': 12})
        plt.title(f"Perception Compute Breakdown\nTotal Latency: {total_latency:.1f}ms ({true_fps:.1f} FPS)", fontsize=14, fontweight='bold')
        plt.axis('equal') 
        plt.tight_layout()
        plt.show()
    else:
        print("\n[WARNING] Not enough frames recorded to generate a chart. Try running it for longer.")