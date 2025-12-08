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
Simple visualization utilities for OSRM routes.
Shows route paths for safety verification.
"""

import logging
import numpy as np
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib not available. Install with: pip install matplotlib")

from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def visualize_route_simple(
    route_coords: List[Tuple[float, float]],
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    home_lat: float,
    home_lon: float,
    title: str = "Route Visualization",
    save_path: str = "route_visualization.png",
    show_immediately: bool = True,
) -> bool:
    """
    Route visualization in local metric coordinates (meters from home position).

    Args:
        route_coords: List of (lat, lon) tuples for the route
        start_lat, start_lon: Starting coordinates
        end_lat, end_lon: Ending coordinates
        home_lat, home_lon: Home reference position
        title: Title for the plot
        save_path: Where to save the image
        show_immediately: Whether to display the plot immediately

    Returns:
        bool: True if visualization was successful
    """
    if not PLOTTING_AVAILABLE:
        logger.error("Matplotlib not available for visualization.")
        return False

    if not route_coords:
        logger.error("No route coordinates provided for visualization")
        return False

    try:
        # Convert all coordinates to local NED frame (meters)
        def gps_to_ned(lat, lon, lat_ref, lon_ref):
            """Convert GPS to local NED coordinates."""
            earth_radius = 6378145.0
            deg_to_rad = 0.01745329252
            Y = earth_radius * (lat - lat_ref) * deg_to_rad
            X = earth_radius * np.cos(lat_ref * deg_to_rad) * (lon - lon_ref) * deg_to_rad
            return X, Y

        # Convert route coordinates to local frame
        route_x, route_y = [], []
        for lat, lon in route_coords:
            x, y = gps_to_ned(lat, lon, home_lat, home_lon)
            route_x.append(x)
            route_y.append(y)

        # Convert start/end to local frame
        start_x, start_y = gps_to_ned(start_lat, start_lon, home_lat, home_lon)
        end_x, end_y = gps_to_ned(end_lat, end_lon, home_lat, home_lon)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot the route in local coordinates
        ax.plot(route_x, route_y, "r-", linewidth=3, label="Route", zorder=2)

        # Add start and end markers
        ax.scatter(
            start_x,
            start_y,
            c="lime",
            s=200,
            marker="o",
            edgecolors="darkgreen",
            linewidth=2,
            label="START",
            zorder=3,
        )
        ax.scatter(
            end_x,
            end_y,
            c="blue",
            s=200,
            marker="s",
            edgecolors="darkblue",
            linewidth=2,
            label="END",
            zorder=3,
        )

        # Calculate bounds with padding in meters
        x_range = max(route_x) - min(route_x)
        y_range = max(route_y) - min(route_y)
        padding = max(x_range, y_range) * 0.1 + 10  # At least 10m padding

        ax.set_xlim(min(route_x) - padding, max(route_x) + padding)
        ax.set_ylim(min(route_y) - padding, max(route_y) + padding)

        # CRITICAL: Set equal aspect ratio AFTER setting limits for proper scaling
        ax.set_aspect("equal", adjustable="box")

        # Labels and formatting - now in meters!
        ax.set_xlabel("East (meters)", fontsize=12)
        ax.set_ylabel("North (meters)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc="upper right")

        # Title with safety warning - show distances in meters
        distance_m = sum(
            np.sqrt((route_x[i + 1] - route_x[i]) ** 2 + (route_y[i + 1] - route_y[i]) ** 2)
            for i in range(len(route_x) - 1)
        )

        ax.set_title(
            f"🚨 SAFETY CHECK: {title} 🚨\n"
            f"Distance: {distance_m:.1f}m | "
            f"START: ({start_x:.1f}, {start_y:.1f})m | "
            f"END: ({end_x:.1f}, {end_y:.1f})m\n"
            f"Reference: ({home_lat:.6f}, {home_lon:.6f})",
            fontsize=14,
            color="darkred",
            weight="bold",
            pad=20,
        )

        # Add safety warning text
        textstr = (
            "⚠️ VERIFY:\n• Route is appropriate\n• Consider traffic/safety\n• Check road conditions"
        )
        props = dict(boxstyle="round", facecolor="yellow", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Route visualization saved to {save_path}")

        if show_immediately:
            # Show the plot
            logger.info("🚨 ROUTE VISUALIZATION DISPLAYED - Close the window when done reviewing!")
            print("\n" + "=" * 60)
            print("🚨 ROUTE SAFETY CHECK WINDOW IS OPEN!")
            print("   Please review the route carefully for safety.")
            print("   Close the window when you're finished reviewing.")
            print("=" * 60)

            plt.show(block=True)  # Block until user closes window
        else:
            plt.close()

        return True

    except Exception as e:
        logger.error(f"Route visualization failed: {e}")
        return False


def plot_route_points(
    route_coords: List[Tuple[float, float]],
    home_lat: float,
    home_lon: float,
    title: str = "Route Points",
) -> bool:
    """
    Scatter plot of route points in local metric coordinates for debugging.

    Args:
        route_coords: List of (lat, lon) tuples
        home_lat, home_lon: Reference position
        title: Plot title

    Returns:
        bool: True if successful
    """
    if not PLOTTING_AVAILABLE or not route_coords:
        return False

    try:
        # Convert to local coordinates
        def gps_to_ned(lat, lon, lat_ref, lon_ref):
            earth_radius = 6378145.0
            deg_to_rad = 0.01745329252
            Y = earth_radius * (lat - lat_ref) * deg_to_rad
            X = earth_radius * np.cos(lat_ref * deg_to_rad) * (lon - lon_ref) * deg_to_rad
            return X, Y

        route_x, route_y = [], []
        for lat, lon in route_coords:
            x, y = gps_to_ned(lat, lon, home_lat, home_lon)
            route_x.append(x)
            route_y.append(y)

        plt.figure(figsize=(10, 8))
        plt.scatter(route_x, route_y, c=range(len(route_x)), cmap="viridis", s=20)
        plt.colorbar(label="Point Index")
        plt.xlabel("East (meters)")
        plt.ylabel("North (meters)")
        plt.title(f"{title} ({len(route_coords)} points)")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        # Annotate start and end
        if len(route_coords) >= 2:
            plt.annotate(
                "START",
                (route_x[0], route_y[0]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="green", alpha=0.7),
            )
            plt.annotate(
                "END",
                (route_x[-1], route_y[-1]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
            )

        plt.tight_layout()
        plt.show()

        return True

    except Exception as e:
        logger.error(f"Point plotting failed: {e}")
        return False
