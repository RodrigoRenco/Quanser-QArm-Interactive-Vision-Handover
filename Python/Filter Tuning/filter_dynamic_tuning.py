import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PARAMETERS
RECORD_DATA = False  # Set to False after recording test
DATA_FILE = r"Python\filter_tuningkalman_tuning_data.csv"
BOTTLE_DIAMETER = 0.065  # Meters, for pinhole depth calculation

R_val = 0.000008 # Measured variance
Q_val = 1e-6   # Tuning parameter

# DATA RECORDING MODE (runs when RECORD_DATA is True)
if RECORD_DATA:
    from ultralytics import YOLO
    import pyrealsense2 as rs

    MODEL_PATH = r"Detection Models\model_bottle.pt"
    MIN_CONFIDENCE = 0.85
    RECORD_SECONDS = 15.0
    DEPTH_KERNEL_SIZE = 5
    half_k = DEPTH_KERNEL_SIZE // 2

    model = YOLO(MODEL_PATH, task='detect')
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    recorded_data = []
    start_time = time.perf_counter()

    print("[INSTRUCTION] Move the bottle around like a real handover. Recording for 15s...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame: continue

            intr = color_frame.profile.as_video_stream_profile().intrinsics
            frame = np.asanyarray(color_frame.get_data())
            results = model(frame, verbose=False)
            detections = results[0].boxes

            sens_z, pinh_z, found = 0.0, 0.0, 0

            if detections is not None and len(detections) > 0:
                det = detections[0]
                if det.conf.item() >= MIN_CONFIDENCE:
                    found = 1
                    xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    u, v = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
                    w_pixels = xmax - xmin

                    # Sensor Depth
                    depths = []
                    for i in range(-half_k, half_k + 1):
                        for j in range(-half_k, half_k + 1):
                            if 0 <= u + i < 640 and 0 <= v + j < 480:
                                d = depth_frame.get_distance(u + i, v + j)
                                if d > 0: depths.append(d)
                    if depths: sens_z = np.median(depths)

                    # Pinhole Depth
                    if w_pixels > 0: pinh_z = (intr.fx * BOTTLE_DIAMETER) / w_pixels

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            recorded_data.append([sens_z, pinh_z, found])

            elapsed = time.perf_counter() - start_time
            cv2.putText(frame, f"RECORDING: {max(0, RECORD_SECONDS - elapsed):.1f}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Dynamic Recording", frame)

            if elapsed >= RECORD_SECONDS or cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    df = pd.DataFrame(recorded_data, columns=['SensZ', 'PinhZ', 'Found'])
    df.to_csv(DATA_FILE, index=False)
    print(f"[SUCCESS] Data saved to {DATA_FILE}. Change RECORD_DATA to False and run again to tune!")

# TUNING & GRAPHING MODE (runs when RECORD_DATA is False)
else:
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("[ERROR] No data found. Set RECORD_DATA = True to record first!")
        exit()

    filtered_z_list = []
    raw_fused_list = []

    # Initialize Kalman
    x_est = df['SensZ'].iloc[0] if df['SensZ'].iloc[0] > 0 else 0.3
    P = 1.0

    consecutive_outliers = 0 # Counter for consecutive outliers
    
    # Simulate the exact Simulink logic
    for i, row in df.iterrows():
        z_sens = row['SensZ']
        z_pinh = row['PinhZ']
        found = row['Found']

        P = P + Q_val  # Prediction

        if found == 1:
            depth = z_sens if z_sens > 0 else z_pinh
            
            # Soft Switch
            if depth < 0.25: w_sens, w_pinh = 0.0, 1.0
            elif depth > 0.35: w_sens, w_pinh = 1.0, 0.0
            else:
                w_sens = (depth - 0.25) / 0.10
                w_pinh = 1.0 - w_sens

            z_fused = (w_sens * z_sens) + (w_pinh * z_pinh)
            
            # Outlier Detection
            distance_jump = abs(z_fused - x_est)
            
            if distance_jump < 0.15: 
                # Normal update
                K = P / (P + R_val)
                x_est = x_est + K * (z_fused - x_est)
                P = (1 - K) * P
                
                consecutive_outliers = 0
                
            else:
                # Detected an outlier
                consecutive_outliers += 1
                
                # Reset logic
                if consecutive_outliers > 5:
                    x_est = z_fused
                    P = 1.0
                    consecutive_outliers = 0
                    
        else:
            z_fused = raw_fused_list[-1] if len(raw_fused_list) > 0 else 0

        raw_fused_list.append(z_fused)
        filtered_z_list.append(x_est)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(raw_fused_list)), raw_fused_list, color='lightblue', label='Raw Fused Camera Data', alpha=0.6)
    plt.plot(filtered_z_list, color='red', linewidth=2.5, label=f'Kalman Filter (Q={Q_val}, R={R_val})')
    plt.title("Offline Kalman Filter Tuning")
    plt.xlabel("Frames")
    plt.ylabel("Z-Depth (meters)")
    plt.legend()
    plt.grid(True)
    plt.show()