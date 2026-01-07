# Manipulation Planning Stack

A Protocol-based motion planning system for robotic manipulators using Drake as the physics backend.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Code                                │
│   Uses Protocol types: WorldSpec, KinematicsSpec, PlannerSpec   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Factory Functions                           │
│   create_world(), create_kinematics(), create_planner()         │
│   Only place that knows about concrete implementations          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  Concrete Implementations                       │
│   DrakeWorld, DrakeKinematics, DrakePlanner, DrakeRRTStar       │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### WorldSpec (Protocol)
The core backend owning physics/collision. Responsibilities:
- Robot and obstacle management
- Collision checking
- Forward kinematics
- Context management for thread safety

Possible implementations:
- `DrakeWorld` (current) - Drake MultibodyPlant + SceneGraph
- `MuJoCoWorld` - MuJoCo physics engine
- `PyBulletWorld` - PyBullet simulation
- `IsaacWorld` - NVIDIA Isaac Sim

### KinematicsSpec (Protocol)
Stateless IK operations. Methods:
- `solve()`: Full optimization-based IK with collision checking
- `solve_iterative()`: Jacobian-based iterative IK
- `solve_differential()`: Single Jacobian step for velocity control

Possible implementations:
- `DrakeKinematics` (current) - Drake's optimization-based IK
- `TracIKKinematics` - Trac-IK solver
- `KDLKinematics` - Orocos KDL
- `PinocchioKinematics` - Pinocchio library

### PlannerSpec (Protocol)
Joint-space path planning. Methods:
- `plan_joint_path()`: Plan collision-free path from start to goal

Possible implementations:
- `DrakeRRTConnect` (current) - Drake's bidirectional RRT
- `OMPLPlanner` - OMPL library (PRM, RRT*, BIT*, etc.)
- `CuRoBoPlanner` - NVIDIA cuRobo GPU-accelerated planning
- `MPNetPlanner` - Neural network motion planning

## Quickstart

### Basic Usage

```python
import numpy as np
from dimos.manipulation.planning import (
    create_world,
    create_kinematics,
    create_planner,
    RobotModelConfig,
    Obstacle,
    ObstacleType,
)

# 1. Create components via factory functions
world = create_world(backend="drake", enable_viz=True)
kinematics = create_kinematics(backend="drake")
planner = create_planner(name="rrt_connect", backend="drake")

# 2. Configure robot
config = RobotModelConfig(
    name="xarm6",
    urdf_path="/path/to/xarm6.urdf",
    base_pose=np.eye(4),
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    end_effector_link="link6",
    base_link="link_base",
)

# 3. Add robot and finalize
robot_id = world.add_robot(config)
world.finalize()

# 4. Add obstacles (can be done after finalize)
obstacle = Obstacle(
    name="table",
    obstacle_type=ObstacleType.BOX,
    pose=np.array([[1,0,0,0.5], [0,1,0,0], [0,0,1,-0.05], [0,0,0,1]]),
    dimensions=(0.6, 0.4, 0.02),
)
world.add_obstacle(obstacle)

# 5. Plan motion
q_start = np.zeros(6)
q_goal = np.array([0.5, 0.3, -0.2, 0.1, 0.4, 0.0])

result = planner.plan_joint_path(world, robot_id, q_start, q_goal, timeout=10.0)
if result.is_success():
    print(f"Found path with {len(result.path)} waypoints")
```

### Using create_planning_stack (Convenience)

```python
from dimos.manipulation.planning import create_planning_stack, RobotModelConfig

# Creates world, kinematics, planner, adds robot, and finalizes in one call
world, kinematics, planner, robot_id = create_planning_stack(
    robot_config=config,
    enable_viz=True,
    planner_name="rrt_connect",
)
```

### Using Contexts for Collision Checking

```python
# Use scratch_context for thread-safe operations
with world.scratch_context() as ctx:
    world.set_positions(ctx, robot_id, q_test)

    if world.is_collision_free(ctx, robot_id):
        ee_pose = world.get_ee_pose(ctx, robot_id)
        print(f"Valid config, EE at: {ee_pose[:3, 3]}")
```

### IK Solving

```python
target_pose = np.eye(4)
target_pose[:3, 3] = [0.3, 0.2, 0.4]  # Position

result = kinematics.solve(
    world=world,
    robot_id=robot_id,
    target_pose=target_pose,
    seed=current_joints,  # Optional initial guess
    check_collision=True,
)

if result.is_success():
    goal_joints = result.joint_positions
```

### Using WorldMonitor for Reactive State Sync

```python
from dimos.manipulation.planning.monitor import WorldMonitor

monitor = WorldMonitor(backend="drake", enable_viz=True)
robot_id = monitor.add_robot(config)
monitor.finalize()
monitor.start_state_monitor(robot_id)

# Feed joint states (e.g., from subscriber)
monitor.on_joint_state(joint_state_msg, robot_id)

# Check collision at configuration
is_valid = monitor.is_state_valid(robot_id, q_test)
```

## Directory Structure

```
planning/
├── spec.py              # Protocol definitions and data classes
├── factory.py           # Factory functions (only file with concrete imports)
├── world/
│   └── drake_world.py   # DrakeWorld implementation
├── kinematics/
│   └── drake_kinematics.py
├── planners/
│   └── drake_planner.py # RRT-Connect and RRT* implementations
├── monitor/
│   ├── world_monitor.py       # WorldMonitor for reactive state sync
│   ├── world_state_monitor.py # Robot state tracking
│   └── world_obstacle_monitor.py
├── trajectory_generator/
│   └── joint_trajectory_generator.py  # Time-parameterized trajectories
├── utils/
│   ├── path_utils.py          # Path interpolation
│   └── kinematics_utils.py    # FK/IK utilities
└── examples/
    ├── planning_tester.py     # Interactive CLI (standalone)
    └── manipulation_client.py # RPC client for deployed modules
```

## Interactive Tools

### Planning Tester (Standalone)

```bash
# Default (Piper robot)
python -m dimos.manipulation.planning.examples.planning_tester

# xArm 6-DOF
python -m dimos.manipulation.planning.examples.planning_tester --robot xarm6

# xArm 7-DOF
python -m dimos.manipulation.planning.examples.planning_tester --robot xarm7
```

Commands:
- `joints` / `j` - Show current joint positions
- `home` - Move to home position
- `random` - Move to random valid configuration
- `ee` - Show end-effector pose
- `collision` - Check current collision status
- `ik <x> <y> <z>` - Solve IK for position
- `plan <j1> <j2> ...` - Plan to joint configuration
- `add` - Add obstacle interactively
- `list` - List all obstacles
- `clear` - Remove all obstacles
- `quit` / `q` - Exit

### Manipulation Client (RPC)

For interacting with a deployed ManipulationModule:

```bash
# Start the manipulation blueprint first
dimos run xarm6-manipulation

# Then run the client
python -m dimos.manipulation.planning.examples.manipulation_client
```

Commands:
- `state`, `ee`, `joints`, `url` - Query state
- `pose x y z [r p y]` - Move to pose
- `plan pose/joint ...` - Plan without executing
- `preview` - Preview path in Meshcat
- `execute` - Execute planned trajectory
- `box`, `sphere`, `remove`, `clear` - Manage obstacles

### Supported Robots

| Robot | DOF | Description |
|-------|-----|-------------|
| `piper` | 6 | Agilex Piper arm (default) |
| `xarm6` | 6 | UFactory xArm 6-DOF |
| `xarm7` | 7 | UFactory xArm 7-DOF |

## Key Design Decisions

1. **Protocol-based architecture**: All code uses Protocol types, enabling future backend swaps
2. **Factory pattern**: Only factory functions know about concrete implementations
3. **Context management**: Thread-safe collision checking via scratch contexts
4. **Dynamic obstacles**: Obstacles can be added/removed after `finalize()`
5. **Stateless IK/Planning**: Kinematics and planners are stateless

## Available Planners

| Planner | Description |
|---------|-------------|
| `rrt_connect` | Bidirectional RRT with greedy connect (fast) |
| `rrt_star` | RRT* with rewiring (asymptotically optimal) |

## Obstacle Types

| Type | Dimensions |
|------|------------|
| `BOX` | (width, height, depth) |
| `SPHERE` | (radius,) |
| `CYLINDER` | (radius, height) |
| `MESH` | mesh_path parameter |
