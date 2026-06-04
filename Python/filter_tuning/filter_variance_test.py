import time
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# SET UP
MODEL_PATH = r"Detection Models\model_bottle.pt"
MIN_CONFIDENCE = 0.85
RECORD_SECONDS = 10.0  # Time to record data after detection
DEPTH_KERNEL_SIZE = 5
half_k = DEPTH_KERNEL_SIZE // 2

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH, task='detect')

print("[INFO] Starting RealSense camera...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

recorded_depths = []
recording_started = False
start_time = 0

print(f"\n[INSTRUCTION] Place the bottle ~40cm away and do not touch it.")
print(f"[INSTRUCTION] Recording will start automatically once detected...\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        
        # Run YOLO
        results = model(frame, verbose=False)
        detections = results[0].boxes

        sensor_distance = 0.0

        if detections is not None and len(detections) > 0:
            # Grab the first valid detection
            for det in detections:
                if det.conf.item() >= MIN_CONFIDENCE:
                    xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    u, v = int((xmin + xmax) / 2), int((ymin + ymax) / 2)

                    # Extract depth as usual
                    depths = []
                    for i in range(-half_k, half_k + 1):
                        for j in range(-half_k, half_k + 1):
                            if 0 <= u + i < 640 and 0 <= v + j < 480:
                                d = depth_frame.get_distance(u + i, v + j)
                                if d > 0:
                                    depths.append(d)

                    if depths:
                        sensor_distance = np.median(depths)
                        
                        # Start the timer on the first successful read
                        if not recording_started:
                            recording_started = True
                            start_time = time.perf_counter()
                            print(f"[STATUS] Target locked. Recording for {RECORD_SECONDS} seconds...")

                        # Only record valid depths while recording
                        if recording_started:
                            recorded_depths.append(sensor_distance)

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    break # Only process one object

        # Calculate time elapsed
        if recording_started:
            elapsed = time.perf_counter() - start_time
            remaining = max(0, RECORD_SECONDS - elapsed)
            
            cv2.putText(frame, f"RECORDING: {remaining:.1f}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Stop condition (after recording time is up)
            if elapsed >= RECORD_SECONDS:
                break
        else:
            cv2.putText(frame, "WAITING FOR TARGET...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Static R-Value Calibration", frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC to quit early
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# CALCULATE R VALUE (VARIANCE)
if len(recorded_depths) > 10:
    z_array = np.array(recorded_depths)
    
    variance = np.var(z_array)
    mean_dist = np.mean(z_array)
    
    print("\n" + "="*50)
    print("CALIBRATION COMPLETE")
    print("="*50)
    print(f"Total frames recorded: {len(z_array)}")
    print(f"Average Distance:      {mean_dist:.4f} meters")
    print(f"Variance (R VALUE):    {variance:.6f}")
    print("="*50)
    print(f"Final Value: R_val = {variance:.6f};\n")
else:
    print("\n[ERROR] Not enough data recorded. Make sure the bottle stays visible.")