#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""
OSRM Router for DIMOS using Open Source Routing Machine.
Fast routing service with pre-processed OSM data.
"""

import logging
import numpy as np
import requests
import time
from typing import Optional, List, Tuple
import math

from dimos.core import Module, Out, In, rpc
from dimos.utils.logging_config import setup_logger
from dimos.robot.outdoor_planner.osrm_visualization import visualize_route_simple

# Import LCM message types
from dimos_lcm.nav_msgs import Path
from dimos_lcm.geometry_msgs import PoseStamped, Point, Quaternion
from dimos_lcm.sensor_msgs import NavSatFix
from dimos_lcm.std_msgs import Header, Time
from nav_msgs.msg import Odometry

logger = setup_logger(__name__)


class OSRMRouter:
    """Simple OSRM router using HTTP API."""

    def __init__(self, profile: str = "cycling", osrm_url: str = None):
        """Initialize OSRM router."""
        self.profile = profile
        self.osrm_url = osrm_url or "http://router.project-osrm.org"

        # Earth constants for coordinate conversion
        self.earthRadius = 6378145.0
        self.DEG_TO_RAD = 0.01745329252

        logger.info(f"OSRM Router: {self.osrm_url} (profile={profile})")
        self._test_connection()

    def _test_connection(self):
        """Quick connection test."""
        try:
            test_url = f"{self.osrm_url}/route/v1/{self.profile}/0,0;1,1?overview=false"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                logger.info("✓ OSRM server ready")
            else:
                logger.warning(f"OSRM server status: {response.status_code}")
        except Exception as e:
            logger.warning(f"OSRM test failed: {e}")

    def find_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        home_lat: float,
        home_lon: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Find route and convert to local NED coordinates."""
        try:
            route_coords = self.get_route_coords(start_lat, start_lon, end_lat, end_lon)
            if not route_coords:
                return None, None

            # Convert to NED coordinates
            lats = np.array([coord[0] for coord in route_coords])
            lons = np.array([coord[1] for coord in route_coords])
            X, Y = self.calcposNED(lats, lons, home_lat, home_lon)

            # Interpolate path for smooth following
            x_interp, y_interp = [], []
            for i in range(len(X) - 1):
                dist = np.linalg.norm([X[i + 1] - X[i], Y[i + 1] - Y[i]])
                N = max(int(dist / 1), 1)  # 1m spacing
                x_interp.append(np.linspace(X[i], X[i + 1], N))
                y_interp.append(np.linspace(Y[i], Y[i + 1], N))

            if x_interp and y_interp:
                X, Y = np.hstack(x_interp), np.hstack(y_interp)

            logger.info(f"✓ Route: {len(route_coords)} points → {len(X)} interpolated")
            return X, Y

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return None, None

    def get_route_coords(
        self, start_lat: float, start_lon: float, end_lat: float, end_lon: float
    ) -> List[Tuple[float, float]]:
        """Get route coordinates from OSRM."""
        try:
            # Build OSRM request
            coords = f"{start_lon},{start_lat};{end_lon},{end_lat}"
            url = f"{self.osrm_url}/route/v1/{self.profile}/{coords}"
            params = {"overview": "full", "geometries": "geojson"}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["code"] != "Ok":
                logger.error(f"OSRM error: {data.get('message', 'Unknown')}")
                return []

            # Extract coordinates: [lon, lat] → (lat, lon)
            geometry = data["routes"][0]["geometry"]["coordinates"]
            return [(coord[1], coord[0]) for coord in geometry]

        except Exception as e:
            logger.error(f"Failed to get route: {e}")
            return []

    def calcposNED(
        self, lat: np.ndarray, lon: np.ndarray, lat_ref: float, lon_ref: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert GPS to local NED coordinates."""
        Y = self.earthRadius * (lat - lat_ref) * self.DEG_TO_RAD
        X = self.earthRadius * np.cos(lat_ref * self.DEG_TO_RAD) * (lon - lon_ref) * self.DEG_TO_RAD
        return X, Y


class OutdoorPathPlanner(Module):
    """Simple OSRM path planner for DIMOS."""

    # Ports
    start_gps: In[NavSatFix] = None
    end_gps: In[NavSatFix] = None
    home_position: In[NavSatFix] = None
    odom_ardupilot: In[Odometry] = None
    odom_vehicle: In[Odometry] = None
    planned_path: Out[Path] = None

    def __init__(self, profile: str = "cycling", osrm_url: str = None, **kwargs):
        """Initialize path planner."""
        super().__init__(**kwargs)
        self.router = OSRMRouter(profile=profile, osrm_url=osrm_url)
        self.current_start = self.current_end = self.current_home = None
        self.current_odom_ardupilot = None
        self.current_odom_vehicle = None
        logger.info(f"PathPlanner ready (profile={profile})")

    @rpc
    def start(self):
        """Start the path planner."""
        if self.start_gps:
            self.start_gps.subscribe(self._on_start_gps)
        if self.end_gps:
            self.end_gps.subscribe(self._on_end_gps)
        if self.home_position:
            self.home_position.subscribe(self._on_home_position)
        if self.odom_ardupilot:
            self.odom_ardupilot.subscribe(self._on_odom_ardupilot)
        if self.odom_vehicle:
            self.odom_vehicle.subscribe(self._on_odom_vehicle)
        logger.info("PathPlanner started")

    def _on_start_gps(self, msg: NavSatFix):
        """Handle start GPS coordinate."""
        self.current_start = msg
        self._try_plan_path()

    def _on_end_gps(self, msg: NavSatFix):
        """Handle end GPS coordinate."""
        self.current_end = msg
        self._try_plan_path()

    def _on_home_position(self, msg: NavSatFix):
        """Handle home position GPS coordinate."""
        self.current_home = msg
        self._try_plan_path()

    def _on_odom_ardupilot(self, msg: Odometry):
        self.current_odom_ardupilot = msg

    def _on_odom_vehicle(self, msg: Odometry):
        self.current_odom_vehicle = msg

    def _try_plan_path(self):
        """Try to plan path if we have all required coordinates."""
        if self.current_start and self.current_end and self.current_home:
            self._plan_and_publish_path()

    def _plan_and_publish_path(self):
        """Plan path and publish to LCM."""
        try:
            # Extract coordinates
            start_lat = self.current_start.latitude
            start_lon = self.current_start.longitude
            end_lat = self.current_end.latitude
            end_lon = self.current_end.longitude
            home_lat = self.current_home.latitude
            home_lon = self.current_home.longitude

            # Plan route (in ardupilot/earth frame, meters)
            X, Y = self.router.find_route(
                start_lat, start_lon, end_lat, end_lon, home_lat, home_lon
            )

            # Transform path to vehicle frame if odometry is available
            X_v, Y_v = self._transform_path_to_vehicle_frame(X, Y)

            if X_v is not None and Y_v is not None:
                # Create Path message
                path = Path()
                path.header = Header()
                path.header.frame_id = "vehicle_odom"
                # path.header.stamp = current time would go here

                # Add poses
                for i in range(len(X_v)):
                    pose = PoseStamped()
                    pose.header = path.header
                    pose.pose.position = Point()
                    pose.pose.position.x = float(X_v[i])
                    pose.pose.position.y = float(Y_v[i])
                    pose.pose.position.z = 2.0  # Default height

                    pose.pose.orientation = Quaternion()
                    pose.pose.orientation.w = 1.0  # Identity quaternion

                    path.poses.append(pose)

                # Publish path
                if self.planned_path:
                    self.planned_path.publish(path)
                    logger.info(f"Published path with {len(path.poses)} poses (vehicle frame)")

            else:
                logger.error("Failed to plan route")

        except Exception as e:
            logger.error(f"Path planning failed: {e}")

    @rpc
    def plan_path(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        home_lat: float,
        home_lon: float,
    ) -> bool:
        """Plan path between coordinates."""
        X, Y = self.router.find_route(start_lat, start_lon, end_lat, end_lon, home_lat, home_lon)
        return X is not None and Y is not None

    @rpc
    def visualize_current_route(
        self, save_image: str = "route_viz.png", show_immediately: bool = True
    ) -> bool:
        """Show route visualization for safety check."""
        if not (self.current_start and self.current_end and self.current_home):
            logger.error("Need start/end/home coordinates")
            return False

        route_coords = self.router.get_route_coords(
            self.current_start.latitude,
            self.current_start.longitude,
            self.current_end.latitude,
            self.current_end.longitude,
        )

        if not route_coords:
            return False

        # Convert route to local NED (meters)
        home_lat = self.current_home.latitude
        home_lon = self.current_home.longitude
        route_x, route_y = [], []
        for lat, lon in route_coords:
            x, y = self.router.calcposNED(np.array([lat]), np.array([lon]), home_lat, home_lon)
            route_x.append(x[0])
            route_y.append(y[0])
        X = np.array(route_x)
        Y = np.array(route_y)

        # Transform to vehicle frame if odometry is available
        X_v, Y_v = self._transform_path_to_vehicle_frame(X, Y)

        # For visualization, convert back to (lat, lon) if needed, or just plot in vehicle frame (meters)
        from dimos.robot.outdoor_planner.osrm_visualization import plt

        if not plt:
            logger.error("Matplotlib not available for visualization.")
            return False

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(X_v, Y_v, "r-", linewidth=3, label="Route", zorder=2)
        ax.scatter(
            X_v[0],
            Y_v[0],
            c="lime",
            s=200,
            marker="o",
            edgecolors="darkgreen",
            linewidth=2,
            label="START",
            zorder=3,
        )
        ax.scatter(
            X_v[-1],
            Y_v[-1],
            c="blue",
            s=200,
            marker="s",
            edgecolors="darkblue",
            linewidth=2,
            label="END",
            zorder=3,
        )
        x_range = max(X_v) - min(X_v)
        y_range = max(Y_v) - min(Y_v)
        padding = max(x_range, y_range) * 0.1 + 10
        ax.set_xlim(min(X_v) - padding, max(X_v) + padding)
        ax.set_ylim(min(Y_v) - padding, max(Y_v) + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East (meters, vehicle frame)", fontsize=12)
        ax.set_ylabel("North (meters, vehicle frame)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc="upper right")
        ax.set_title(
            f"🚨 SAFETY CHECK: OSRM {self.router.profile.title()} Route (Vehicle Frame) 🚨\n"
            f"START: ({X_v[0]:.1f}, {Y_v[0]:.1f})m | END: ({X_v[-1]:.1f}, {Y_v[-1]:.1f})m",
            fontsize=14,
            color="darkred",
            weight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(save_image, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Route visualization saved to {save_image}")
        if show_immediately:
            plt.show(block=True)
        else:
            plt.close()
        return True

    def _get_2d_pose(self, odom_msg):
        """Extract (x, y, yaw) from nav_msgs/Odometry."""
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        # Yaw from quaternion
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return pos.x, pos.y, yaw

    def _compute_2d_transform(self):
        """Compute 2D transform (Δx, Δy, Δθ) from ardupilot odom to vehicle odom."""
        if not (self.current_odom_ardupilot and self.current_odom_vehicle):
            return None
        x_ap, y_ap, yaw_ap = self._get_2d_pose(self.current_odom_ardupilot)
        x_v, y_v, yaw_v = self._get_2d_pose(self.current_odom_vehicle)
        dx = x_v - x_ap
        dy = y_v - y_ap
        dtheta = yaw_v - yaw_ap
        return dx, dy, dtheta

    def _transform_path_to_vehicle_frame(self, X, Y):
        """Transform arrays of (X, Y) in ardupilot frame to vehicle frame using latest odom transform."""
        tf = self._compute_2d_transform()
        if tf is None:
            logger.warning("No odom transform available; returning original path")
            return X, Y
        dx, dy, dtheta = tf
        cos_theta = math.cos(dtheta)
        sin_theta = math.sin(dtheta)
        X_v = cos_theta * X - sin_theta * Y + dx
        Y_v = sin_theta * X + cos_theta * Y + dy
        return X_v, Y_v


def main():
    """OSRM router demo."""
    # Test coordinates
    start_lat, start_lon = 37.780930, -122.406752
    end_lat, end_lon = 37.786742, -122.405592
    home_lat, home_lon = 37.780930, -122.406752

    print("🚀 OSRM Router Demo")
    print("=" * 30)

    router = OSRMRouter(profile="cycling")

    # Test routing
    start_time = time.time()
    X, Y = router.find_route(start_lat, start_lon, end_lat, end_lon, home_lat, home_lon)
    total_time = time.time() - start_time

    if X is not None and Y is not None:
        print(f"✓ Route found in {total_time:.2f}s")
        print(f"  Points: {len(X)}")
        print(f"  Distance: {np.sum(np.sqrt(np.diff(X) ** 2 + np.diff(Y) ** 2)):.1f}m")

        # Test visualization
        route_coords = router.get_route_coords(start_lat, start_lon, end_lat, end_lon)
        if route_coords:
            print(f"  Visualization: {len(route_coords)} route points")
            viz_success = visualize_route_simple(
                route_coords,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                home_lat,
                home_lon,
                title="OSRM Demo Route",
                show_immediately=True,
            )
            print(
                f"  {'✓' if viz_success else '✗'} Visualization {'shown' if viz_success else 'failed'}"
            )

        print("\n⚠️ Review route for safety before use!")
    else:
        print("✗ Route failed - check connection/coordinates")

    print(f"\n🏁 Done in {total_time:.2f}s")


if __name__ == "__main__":
    main()
