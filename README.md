# SwarmAero 🚁

**Autonomous 3-drone swarm with quaternion EKF state estimation, LQR attitude control, minimum-snap trajectory generation, and graph Laplacian formation control.**

> Built on ROS2 Humble + PX4 SITL + Gazebo | Author: Maanav Shah | Purdue ECE

---

## System Architecture
IMU (250Hz) ──→ Quaternion EKF ──→ State Estimate
GPS  (30Hz) ──→      │                    │
Gazebo GT   ──→ NIS/NEES Validation       │
↓
LQR Attitude Controller
│
↓
Minimum-Snap Trajectory Follower
│
↓
Graph Laplacian Formation Control (3 agents)
---

## Modules

### 1. Quaternion EKF (`src/swarm_ekf/`)
- **State vector**: `[pos(3), vel(3), quat(4), accel_bias(3), gyro_bias(3)]` — 16 states
- Exact discrete quaternion update: `q_{k+1} = q_k ⊗ exp(0.5 * ω * dt)`
- Unit quaternion re-normalization after every prediction step
- Gyro + accel bias modeled as random walk with tuned process noise
- Joseph form covariance update: `P = (I-KH)P(I-KH)^T + KRK^T`
- **NIS/NEES within 95% chi-squared bounds confirming filter consistency**

### 2. LQR Attitude Controller *(in progress)*
### 3. Minimum-Snap Trajectory Generation *(in progress)*
### 4. Multi-Agent Formation Control *(in progress)*

---

## Results

### NIS/NEES Filter Consistency
![NIS/NEES](results/nis_nees.png)

---

## How to Run

**Terminal 1 — PX4 SITL + Gazebo:**
```bash
cd ~/PX4-Autopilot && make px4_sitl gz_x500
```

**Terminal 2 — ROS2 Bridge:**
```bash
MicroXRCEAgent udp4 -p 8888
```

**Terminal 3 — EKF Node:**
```bash
source ~/SwarmAero/ros2_ws/install/setup.bash
ros2 run swarm_ekf ekf_node
```

Hit `Ctrl+C` to stop and auto-save NIS/NEES plots to `results/`.

---

## Dependencies
- ROS2 Humble
- PX4-Autopilot (SITL)
- Gazebo 8
- micro-XRCE-DDS-Agent
- Python: numpy, scipy, matplotlib, transforms3d
