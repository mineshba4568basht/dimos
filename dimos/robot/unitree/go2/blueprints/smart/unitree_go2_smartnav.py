#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Go2 SmartNav blueprint: PGO + ScanCorrector + VoxelMapper + CostMapper + Planner.

Uses PGO for loop-closure-corrected odometry. ScanCorrector re-registers the
robot's raw lidar using PGO's corrected pose and overlays it onto PGO's global
static map via z-column clearing. VoxelMapper consumes the combined output.

Data flow:
    GO2Connection.lidar  (remapped → registered_scan) → PGO + ScanCorrector
    GO2Connection.odom   (remapped → raw_odom)         → PGO + ScanCorrector
    PGO.odom             (corrected PoseStamped)        → ScanCorrector
    PGO.global_static_map                               → ScanCorrector
    ScanCorrector.corrected_map → VoxelGridMapper → CostMapper → ReplanningAStarPlanner
    ReplanningAStarPlanner.cmd_vel → GO2Connection
"""

from dimos.core.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.navigation.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.navigation.loop_closure.pgo import PGO
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.navigation.scan_corrector import ScanCorrector
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.robot.unitree.go2.connection import GO2Connection

unitree_go2_smartnav = (
    autoconnect(
        unitree_go2_basic,
        PGO.blueprint(),
        ScanCorrector.blueprint(),
        VoxelGridMapper.blueprint(voxel_size=0.1),
        CostMapper.blueprint(),
        ReplanningAStarPlanner.blueprint(),
        WavefrontFrontierExplorer.blueprint(),
    )
    .global_config(n_workers=9, robot_model="unitree_go2")
    .remappings(
        [
            (GO2Connection, "lidar", "registered_scan"),
            (GO2Connection, "odom", "raw_odom"),
            (VoxelGridMapper, "lidar", "corrected_map"),
        ]
    )
)

__all__ = ["unitree_go2_smartnav"]
