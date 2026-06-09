# Vision-Based Human-Robot Handover Using a 4-DOF Robotic Manipulator

This repository contains the end-to-end multi-modal perception and control framework for executing vision-guided object detection, hybrid tracking, and collaborative human-robot handover tasks using a 4-DOF Quanser QArm and an Intel RealSense D415 RGB-D camera.

The architecture utilizes a decoupled design approach: a high-speed Python layer handles real-time neural network inference (YOLOv8 object tracking + MediaPipe hand gestures), while a deterministic MATLAB/Simulink layer executes finite state machine logic and joint-space inverse kinematics over a real-time UDP socket loop.

---

## Repository Structure

```text
├── Detection Models/
│   ├── model_bottle.pt              # Custom YOLOv8 weights optimized for bottle tracking
│   └── model_ball.pt                # Custom YOLOv8 weights optimized for tennis ball tracking
├── MATLAB-Simulink Files/
│   └── connecttoUDP_final.slx       # Central control block diagram executing State Machine & IK
├── Python/
│   ├── vision_raw_udp_HD_bottle.py  # Production perception script for the bottle scenario
│   ├── vision_raw_udp_HD_ball.py    # Production perception script for the ball scenario
│   └── hands_detector.py            # Geometric rule-based hand landmark gesture classification
├── Simulation/                      # Offline simulation and trajectory validation tools
│   ├── app1_GUI_trajectoryplanning.mlapp # 3D workspace simulator for polynomial interpolations
│   ├── potentialfield.mlapp         # 3D simulator for dynamic Artificial Potential Fields (APF)
│   ├── trajectories_comparison.m    # MATLAB script evaluating Cartesian profile trade-offs
│   └── different_grafes.m           # MATLAB script generating joint-space dynamic comparison plots
├── .gitignore                       # Configured to automatically ignore Simulink cache artifacts
└── README.md                        # Project documentation and platform installation steps
```
## System Requirements & Prerequisites

To execute this project successfully on physical hardware, make sure to follow these specification baselines:

### 1. Control Layer Environment
* **MATLAB / Simulink:** Version **2025a**.
* **Toolboxes:** DSP System Toolbox is required for high-speed binary socket unpacking operations.
* **Hardware Compiler:** **Quanser QUARC** real-time control package must be configured for local QArm target hardware deployment.

### 2. Perception Layer Environment
* **Python Runtime:** Version **3.12**.
* **Core Dependencies:** Install required wheels inside a clean virtual environment:
  ```bash
  pip install ultralytics pyrealsense2 opencv-python mediapipe numpy pandas matplotlib
  ```

### 3. Hardware Setup Context
* **Actuation System:** Quanser QArm serial manipulator.
* **Sensor:** Intel RealSense D415 camera positioned securely to observed the workspace. Ensure a high-bandwidth USB 3.0 connection and consistent ambient lighting.

## Execution & Startup Sequence

> **Order of Execution Matters:** To prevent local UDP network socket collision or connection binding faults, you must execute the Python vision node **before** starting the Simulink controller.

### Step 1: Initialize the Perception Pipeline (Python)
Navigate to your local repository clone via your command line interface and launch the perception script corresponding to your target object:

* **For the Bottle Sorting/Handover Scenario:**
  ```bash
  python Python/vision_raw_udp_HD_bottle.py
  ```
* **For the Tennis Ball Tracking/Handover Scenario:**
  ```bash
  python Python/vision_raw_udp_HD_ball.py
  ```
A local OpenCV window will appear on-screen, showing the active camera stream, tracking bounding boxes, FPS metrics, and classified MediaPipe gesture variables. The module will immediately begin broadcasting packed 3D spatial coordinate vectors through UDP communication.

### Step 2: Build and Run the Control Loop (Simulink)
* Launch MATLAB 2025a and open the file located at `MATLAB-Simulink Files/connecttoUDP_final.slx.`

* Verify model parameters are properly targeted to your active physical QArm workstation.

* On the Simulink toolbar interface, click the Hardware tab.

* Click Monitor/Tune to compile the state machine layout and push the real-time commands directly onto the physical hardware.

## System Operational State Overview

Once compiled, the system operates completely autonomously according to a deterministic state logic managed inside the central control loop:

1. **Initialization and Search Mode (State 2):** Upon startup, the robot moves smoothly to a safe resting home position `[0.25, 0.0, 0.50]`. After locking spatial stability, it executes a horizontal sinusoidal sweep to scan the workspace until an object is detected and verified through a visual debounce barrier.
2. **Target Tracking Phase (State 0):** Upon confirmed visual contact, the robot actively tracks the target's coordinates. Coordinates pass through a custom 3D Kalman Filter paired with an automated ± 0.15 m Outlier Rejection Gate to isolate the trajectory loop from noise, packet drops, or glare glitches. If the camera loses sight of the object for more than 2.5 seconds, a watchdog timer safely aborts the mission and returns the system to search mode.
3. **Commit Phase (State 1):** When the target object is confirmed within grasping range, the state transitions to execute the approach, grasp, and delivery sequence. An internal trajectory stabilization timer triggers a persistent gripper latching sequence, commanding the claw to close tightly and hold the payload securely while traveling.
4. **Interactive Delivery Mode (State 3):** Serving as the primary human-robot interaction phase, the robot returns to the base coordinates and awaits human instruction. Continuous inputs from hand tracking (LEFT/RIGHT) incrementally modify the target Y-coordinate to translate discrete gestures into fluid, physical displacement. Once a "STOP" gesture is maintained for 0.5 seconds, the final coordinates are registered in memory, triggering the delivery advancement, payload release, and home recovery loops.

## Offline Simulation & Trajectory Validation

Before deploying target profiles onto the physical hardware, the system's underlying trajectory planning laws were modeled, simulated, and benchmarked qualitatively and quantitatively. These scripts and graphical user interfaces are located within the `Simulation/` directory.

### 1. Interactive 3D Kinematic simulators (.mlapp)
Open MATLAB and navigate to the `Simulation/` directory to launch either application via the App Designer environment:

* **Polynomial Interpolation Simulator (`app1_GUI_trajectoryplanning.mlapp`):** Allows users to input a custom Cartesian coordinate target and visually analyze the arm's path tracking response. The simulator calculates the analytical inverse kinematics and allows side-by-side selection of four discrete point-to-point interpolation laws: **Linear (LERP)**, **Cubic Spline**, **Quintic Spline**, and **Trapezoidal** profiles.
* **Real-Time Potential Field Simulator (`potentialfield.mlapp`):** Simulates a dynamic tracking environment where the target evolves continuously in time rather than remaining static. This app models the attractive virtual forces driving the end-effector to an expected interception point while enforcing joint boundaries and mechanical workspace constraints.

### 2. Quantitative Performance Scripting (.m)
To generate the exact dynamic profiles evaluating structural stress and velocity profiles, execute the standalone comparison scripts in the MATLAB command terminal:

* **`trajectories_comparison.m`:** Computes and plots the continuous time evolution of Cartesian position coordinates ($X, Y, Z$) to directly compare path directness between the cubic spline, vector step, and potential field configurations.
* **`different_grafes.m`:** Generates an engineering matrix tracking position, velocity, and acceleration joint-by-joint. This script was utilized to validate the choice of the potential field method by identifying velocity phase lags and acceleration discontinuities across the competing mathematical control laws.

## Project Team & Contributors
* **Lorenza D'Amario** – Trajectory Generation and Control Module
* **Ana Koenig** – Trajectory Generation and Control Module
* **Emanoel Victor Da Silva Morais** – State Machine Module
* **Rodrigo Rencoret** – Object and Hand Gesture Detection Module
* **Martín Schulz** – Image Processing and Depth Perception Module 

**Supervised by:** Etienne Chassaing & Maria Makarov (CentraleSupélec Robotics Project Cluster, P19).
