# Go2 Trained Policy - Deployment Guide

## Trained Policy Files

| File | Path |
|------|------|
| ONNX policy | `unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-03-17_16-18-21/exported/policy.onnx` |
| JIT policy | `unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-03-17_16-18-21/exported/policy.pt` |
| Deploy config | `unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-03-17_16-18-21/params/deploy.yaml` |
| Env config | `unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-03-17_16-18-21/params/env.yaml` |
| Agent config | `unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-03-17_16-18-21/params/agent.yaml` |
| PT checkpoints | `unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-03-17_16-18-21/model_*.pt` |

## Policy Specs

- **Input:** 45-dim observation vector
- **Output:** 12-dim action vector (joint position offsets)
- **Network:** MLP [45 -> 512 -> 256 -> 128 -> 12] with ELU activations
- **Control rate:** 50 Hz (step_dt = 0.02s)

## Robot Configuration

- **Robot:** Unitree Go2 (12 DoF quadruped)
- **PD gains:** kp = 25.0, kd = 0.5 (all joints)
- **Action scale:** 0.25 (policy_output * 0.25 + default_pos = target joint pos)
- **Joint ID mapping (Isaac Lab -> robot):** `[3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]`
- **Default joint positions (rad):**
  ```
  [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
  ```
  Order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf

## Observation Vector (45 dims)

Build this vector at each control step (50 Hz):

| Index | Name | Dims | Scale | Source |
|-------|------|------|-------|--------|
| 0-2 | base_ang_vel | 3 | 0.2 | IMU gyroscope (body frame) |
| 3-5 | projected_gravity | 3 | 1.0 | Gravity vector in body frame (from IMU quaternion) |
| 6-8 | velocity_commands | 3 | 1.0 | Desired [vx, vy, yaw_rate] |
| 9-20 | joint_pos_rel | 12 | 1.0 | current_joint_pos - default_joint_pos |
| 21-32 | joint_vel_rel | 12 | 0.05 | current_joint_vel * 0.05 |
| 33-44 | last_action | 12 | 1.0 | Previous policy output (raw, before scaling) |

### Projected Gravity Calculation

```python
from scipy.spatial.transform import Rotation
quat = [w, x, y, z]  # from IMU
R = Rotation.from_quat([x, y, z, w]).as_matrix()
projected_gravity = R.T @ [0, 0, -1]
```

## Action Processing

```python
raw_action = policy(obs)              # 12 values from ONNX
target_pos = raw_action * 0.25 + default_joint_pos
# Clip: [-100, 100] (effectively no clip)
# Send to robot with PD: torque = kp * (target_pos - current_pos) + kd * (0 - current_vel)
```

## Minimal Python Integration

```python
import numpy as np
import onnxruntime as ort

# Load policy
session = ort.InferenceSession("policy.onnx")

# Constants
DEFAULT_POS = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
JOINT_MAP = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]  # isaac_lab_order -> robot_order
ACTION_SCALE = 0.25
KP = 25.0
KD = 0.5
DT = 0.02  # 50 Hz

last_action = np.zeros(12)

def step(imu_gyro, imu_quat, velocity_cmd, joint_pos, joint_vel):
    """
    Args:
        imu_gyro: [wx, wy, wz] angular velocity in body frame
        imu_quat: [w, x, y, z] orientation quaternion
        velocity_cmd: [vx, vy, yaw_rate] desired velocity
        joint_pos: [12] current joint positions (Isaac Lab order)
        joint_vel: [12] current joint velocities (Isaac Lab order)
    Returns:
        target_pos: [12] joint position targets (robot order, use JOINT_MAP)
        kp, kd: PD gains
    """
    global last_action

    # Projected gravity
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    proj_grav = R.T @ np.array([0, 0, -1])

    # Build observation
    obs = np.concatenate([
        imu_gyro * 0.2,                    # 3
        proj_grav,                          # 3
        velocity_cmd,                       # 3
        joint_pos - DEFAULT_POS,            # 12
        joint_vel * 0.05,                   # 12
        last_action,                        # 12
    ]).astype(np.float32).reshape(1, -1)

    # Run policy
    action = session.run(None, {"obs": obs})[0][0]
    last_action = action.copy()

    # Convert to joint targets
    target_pos = action * ACTION_SCALE + DEFAULT_POS

    # Map to robot joint order
    robot_targets = np.zeros(12)
    for i, robot_idx in enumerate(JOINT_MAP):
        robot_targets[robot_idx] = target_pos[i]

    return robot_targets, KP, KD
```

## Sim2Sim Testing (MuJoCo)

### Start MuJoCo sim
```bash
cd ~/Documents/learning/unitree_mujoco/simulate/build
./unitree_mujoco
```

### Start Go2 controller
```bash
cd ~/Documents/learning/unitree_rl_lab/deploy/robots/go2/build
./go2_ctrl
```

### Controls (Xbox gamepad)
1. **LT + A** -> Stand up (FixStand mode)
2. **Start (menu button)** -> Run policy (Velocity mode)
3. **Left stick** -> Velocity commands
4. **LT + B** -> Back to passive

## Sim2Real

Same controller, just specify the network interface:
```bash
./go2_ctrl --network eth0
```

## Training Results (10k iterations)

| Metric | Value |
|--------|-------|
| Mean reward | 31.21 |
| Episode length | 991/1000 |
| Velocity tracking (xy) | 0.16 error |
| Velocity tracking (yaw) | 0.18 error |
| Action std | 0.20 |
| Fall rate | < 0.4% |
