import os
import socket
import struct
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# Setup global parameters 
MODEL_PATH = r"C:\Users\piopi\Desktop\Centrale\PP S8 Robotique\codes\catching the ball\my_model.pt"
MIN_CONFIDENCE = 0.85   # Minimum confidence threshold for YOLO detections  
BALL_DIAMETER = 0.067   # 6.7 cm for a regular tennis ball
BALL_RADIUS = BALL_DIAMETER / 2
DEPTH_KERNEL_SIZE = 5   # Size of the kernel to average depth values
half_k = DEPTH_KERNEL_SIZE // 2

# UDP configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Loading YOLO model
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: The model file was not found : {MODEL_PATH}")
    sys.exit(0)

model = YOLO(MODEL_PATH, task='detect')
print("YOLO model loaded.")

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
print("[INFO] RealSense pipeline started...")

# Main loop
try:
    while True:
        t_start = time.perf_counter()
        
        # Reset payload variables for this frame
        point_sensor = [0.0, 0.0, 0.0]
        point_pinhole = [0.0, 0.0, 0.0]
        found_now = False

        # 1. Capture frames from RealSense
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        frame = np.asanyarray(color_frame.get_data())

        # 2. YOLO Detection
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # 3. Process detections
        if detections is not None:
            for det in detections:
                conf = det.conf.item()
                if conf < MIN_CONFIDENCE:
                    continue

                # 3.1 Get bounding box and calculate center
                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                u = int((xmin + xmax) / 2)
                v = int((ymin + ymax) / 2)
                w_pixels = xmax - xmin  # Width of the bounding box in pixels

                # 3.2 Obtain depth from RealSense at the center
                depths = []
                for i in range(-half_k, half_k + 1):
                    for j in range(-half_k, half_k + 1):
                        if 0 <= u + i < 640 and 0 <= v + j < 480:
                            d = depth_frame.get_distance(u + i, v + j)
                            if d > 0:
                                depths.append(d)
                
                sensor_distance = np.median(depths) if depths else 0.0

                # 3.3 Calculate Pinhole Distance
                pinhole_distance = 0.0
                if w_pixels > 0:
                    fx = intr.fx
                    pinhole_distance = (fx * BALL_DIAMETER) / w_pixels

                # 3.4 Calculate both Raw 3D points
                point_sensor = [0.0, 0.0, 0.0]
                point_pinhole = [0.0, 0.0, 0.0]

                if sensor_distance > 0:
                    point_sensor = rs.rs2_deproject_pixel_to_point(intr, [u, v], sensor_distance + BALL_RADIUS)
                
                if pinhole_distance > 0:
                    point_pinhole = rs.rs2_deproject_pixel_to_point(intr, [u, v], pinhole_distance)

                # Determine if we have at least one valid detection
                if sensor_distance > 0 or pinhole_distance > 0:
                    found_now = True

                # 4. Draw visualizations for debugging
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                # Display both distances on the screen
                dist_str = f"S:{sensor_distance:.2f}m | P:{pinhole_distance:.2f}m"
                # Dinamically calculate text position to avoid going off-screen
                (text_width, text_height), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_x = xmax + 5 # Default position: right side
                # If text goes off the right edge (640px), flip it to the left side of the box
                if text_x + text_width > 640:
                    text_x = xmin - 5 - text_width
                    # If the box is so huge the text still clips on the left, lock it to the screen edge
                    if text_x < 0:
                        text_x = 5
                cv2.putText(frame, dist_str, (text_x, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                break # Only process the first detected ball

        # 5. UDP Communication
        try:
            found_now_flag = 1.0 if found_now else 0.0
            message_bytes = struct.pack(
                'fffffff', # 7 floats: Sensor XYZ, Pinhole XYZ, Detection Flag
                float(point_sensor[0]),  # 1: Sensor X
                float(point_sensor[1]),  # 2: Sensor Y
                float(point_sensor[2]),  # 3: Sensor Z
                float(point_pinhole[0]), # 4: Pinhole X
                float(point_pinhole[1]), # 5: Pinhole Y
                float(point_pinhole[2]), # 6: Pinhole Z
                found_now_flag           # 7: 1.0 if Detected, 0.0 if Lost
            )
            sock.sendto(message_bytes, (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"UDP Error : {e}")

        # 6. Display FPS and Status (Optimized for minimum latency)
        fps = 1.0 / (time.perf_counter() - t_start)
        cv2.putText(frame, f"Raw FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_txt = "SENDING DUAL RAW DATA" if found_now else "TARGET LOST (SENDING 0s)"
        status_col = (0, 255, 0) if found_now else (0, 0, 255)
        cv2.putText(frame, status_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)
        
        # Uncomment these lines to display the individual sensor and pinhole distances on the frame for debugging
        # cv2.putText(frame, f"Sens: [{point_sensor[0]:.2f}, {point_sensor[1]:.2f}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # cv2.putText(frame, f"Pinh: [{point_pinhole[0]:.2f}, {point_pinhole[1]:.2f}]", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("YOLO Ball Detection", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

finally:
    print("\nClosing...")
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
    print("Done.")