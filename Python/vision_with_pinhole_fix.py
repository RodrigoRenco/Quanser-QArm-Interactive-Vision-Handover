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
MIN_CONFIDENCE = 0.85

BALL_DIAMETER = 0.067      # 6.7 cm for a regular tennis ball
BALL_RADIUS = BALL_DIAMETER / 2
COMMIT_DISTANCE = 0.18     # 18 cm: Commit Point (to close the gripper)
DEPTH_KERNEL_SIZE = 5      # Size of the kernel to average depth values around the detected point
GRIP_HOLD = 1.0            # Time in seconds to hold the grip after closing

LOST_TIMEOUT = 0.5         # Time in seconds to keep sending the last known point after losing detection
GRIP_DELAY = 0.75          # Time in seconds to wait in the trigger zone before closing the gripper

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

# Initialize loop variables
last_valid_point_3d = [0.0, 0.0, 0.0]
grip_state = 0
grip_release_time = None
frame_rate_buffer = []
fps_avg_len = 30

last_detection_time = 0.0
grip_trigger_start_time = None

# Main loop
try:
    while True:
        t_start = time.perf_counter()
        current_time = time.time()

        point_to_send = [0.0, 0.0, 0.0]
        found_now = False
        distance = 0.0
        is_estimated = False

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
                w_pixels = xmax - xmin  # Width of the tennis ball in pixels

                # 3.2 Obtain depth from RealSense at the center of the detected object
                half_k = DEPTH_KERNEL_SIZE // 2
                depths = []
                for i in range(-half_k, half_k + 1):
                    for j in range(-half_k, half_k + 1):
                        # Avoid going out of bounds
                        if 0 <= u + i < 640 and 0 <= v + j < 480:
                            d = depth_frame.get_distance(u + i, v + j)
                            if d > 0:
                                depths.append(d)
                
                sensor_distance = np.median(depths) if depths else 0.0

                # 3.3 Pinhole Model Estimation for Close Objects (<30cm)
                if sensor_distance > 0:
                    distance = sensor_distance
                    is_estimated = False
                elif w_pixels > 0:
                    # Use pinhole camera model only for close objects where depth sensor fails
                    fx = intr.fx
                    distance = (fx * BALL_DIAMETER) / w_pixels
                    is_estimated = True
                else:
                    distance = 0.0

                # 3.4 If we have a valid distance, calculate 3D point
                if distance > 0:
                    if not is_estimated:
                        # Adjust Z to point to the center of the ball (since depth points to the surface)
                        distance_to_center = distance + BALL_RADIUS
                    else:
                        # Pinhole already estimates distance to the center based on total width
                        distance_to_center = distance

                    # Calculate X, Y, Z based on the distance to the center
                    point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], distance_to_center)
                     
                    # if distance > 0:
                    # point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], distance)
                    
                    # if not is_estimated:
                        # Adjust Z to point to the center of the ball (since depth points to the surface)
                    #    point_3d[2] += BALL_RADIUS
                    # For estimated points, we assume the center is correct since the pinhole model is based on the ball's diameter
                    last_valid_point_3d = point_3d
                    point_to_send = point_3d
                    found_now = True
                    last_detection_time = current_time # Update last detection time

                    # 3.5 Delayed Gripper Actuation Logic
                    if distance <= COMMIT_DISTANCE:
                        if grip_trigger_start_time is None:
                            grip_trigger_start_time = current_time # Start the 0.75s timer
                        elif current_time - grip_trigger_start_time >= GRIP_DELAY:
                            grip_state = 1
                            grip_release_time = current_time + GRIP_HOLD
                    else:
                        grip_trigger_start_time = None # Reset timer if the ball moves away


                # 4. Draw bounding box, center, and distance on the frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                
                dist_str = f"{distance:.2f}m {'(E)' if is_estimated else ''}"
                color_text = (0, 165, 255) if is_estimated else (0, 255, 255) # Orange for estimated, Yellow for valid sensor reading
                cv2.putText(frame, dist_str, (xmax + 5, ymin + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
                
                break # Only process the first detected ball
        
        # 5. Handle grip release timing
        if grip_release_time is not None:
            if current_time >= grip_release_time:
                grip_state = 0
                grip_release_time = None
            else:
                grip_state = 1
            
        # 6. Null Coordinate Fix & Gripper Reset
        if not found_now:
            grip_trigger_start_time = None # Cancel grip timer if we lose sight of the ball
            
            # Keep sending the last known point for a short duration after losing detection
            if current_time - last_detection_time <= LOST_TIMEOUT:
                point_to_send = last_valid_point_3d
                cv2.putText(frame, f"COASTING ({LOST_TIMEOUT}s Hold)", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                point_to_send = [0.0, 0.0, 0.0]

        try:
            # Convert the point and grip state to bytes and send via UDP
            is_detected_flag = 1.0 if found_now else 0.0
            message_bytes = struct.pack(
                'fffff', # 5 floats: X, Y, Z, Grip State, Detection Flag
                point_to_send[0],
                point_to_send[1],
                point_to_send[2],
                float(grip_state), # Grip state as float (0.0 or 1.0)
                is_detected_flag   # Detection flag as float (0.0 for no detection, 1.0 for detection)
            )
            sock.sendto(message_bytes, (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"Erreur UDP : {e}")

        # 7. Calculate and display FPS and other info
        t_stop = time.perf_counter()
        fps = 1.0 / (t_stop - t_start)
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_fps = np.mean(frame_rate_buffer)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"XYZ: {[round(x,2) for x in point_to_send]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        grip_color = (0, 255, 0) if grip_state == 1 else (0, 0, 255)
        cv2.putText(frame, f"GRIP: {int(grip_state)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, grip_color, 2)

        cv2.imshow("YOLO Ball Detection & Pinhole Hybrid", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

finally:
    print("\nFermeture des flux...")
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
    print("Terminé.")
