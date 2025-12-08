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
Visualization utilities for outdoor path planning.
Displays route maps for safety verification.
"""

import logging
from typing import Optional

try:
    import osmnx as ox
    import networkx as nx
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    ox = None
    nx = None
    plt = None
    PLOTTING_AVAILABLE = False
    logging.warning(
        "Visualization dependencies not available. Install with: pip install matplotlib"
    )

from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def visualize_route(
    graph,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    place: str,
    save_path: str = "route_visualization.png",
    show_immediately: bool = True,
) -> bool:
    """
    Visualize a planned route on the street network - shows immediately for safety!

    Args:
        graph: OSMnx graph object
        start_lat, start_lon: Starting coordinates
        end_lat, end_lon: Ending coordinates
        place: Place name for the route
        save_path: Where to save the image
        show_immediately: Whether to display the plot immediately

    Returns:
        bool: True if visualization was successful
    """
    if not PLOTTING_AVAILABLE or graph is None:
        logger.error(
            "Matplotlib not available for visualization. Install with: pip install matplotlib"
        )
        return False

    try:
        # Get nearest nodes to start/end coordinates
        orig = ox.distance.nearest_nodes(graph, X=start_lon, Y=start_lat)
        dest = ox.distance.nearest_nodes(graph, X=end_lon, Y=end_lat)

        # Find shortest path
        route = nx.shortest_path(graph, orig, dest, weight="length")

        # Get route coordinates for zoom calculation
        route_coords = []
        for node in route:
            node_data = graph.nodes[node]
            route_coords.append((node_data["x"], node_data["y"]))  # lon, lat

        # Calculate route bounds for zooming
        route_lons = [coord[0] for coord in route_coords]
        route_lats = [coord[1] for coord in route_coords]

        # Add padding around the route (about 10% extra on each side)
        lon_padding = (max(route_lons) - min(route_lons)) * 0.1
        lat_padding = (max(route_lats) - min(route_lats)) * 0.1

        # Ensure minimum padding for very short routes
        min_padding = 0.002  # about 200m
        lon_padding = max(lon_padding, min_padding)
        lat_padding = max(lat_padding, min_padding)

        zoom_bounds = {
            "west": min(route_lons) - lon_padding,
            "east": max(route_lons) + lon_padding,
            "south": min(route_lats) - lat_padding,
            "north": max(route_lats) + lat_padding,
        }

        # Create the plot with large, visible elements - zoomed to route area
        fig, ax = ox.plot_graph_route(
            graph,
            route,
            route_linewidth=8,  # Thick route line
            route_color="red",
            node_size=0,
            bgcolor="white",
            edge_color="lightgray",
            edge_linewidth=1,
            figsize=(16, 12),  # Large figure
            show=False,
            close=False,
            bbox=(
                zoom_bounds["north"],
                zoom_bounds["south"],
                zoom_bounds["east"],
                zoom_bounds["west"],
            ),  # N, S, E, W
        )

        # Add highly visible start and end markers
        ax.scatter(
            start_lon,
            start_lat,
            c="lime",
            s=400,
            marker="o",
            edgecolors="darkgreen",
            linewidth=3,
            label="START",
            zorder=10,
        )
        ax.scatter(
            end_lon,
            end_lat,
            c="blue",
            s=400,
            marker="s",
            edgecolors="darkblue",
            linewidth=3,
            label="END",
            zorder=10,
        )

        # Add prominent safety warning and legend
        ax.legend(fontsize=16, loc="upper right", frameon=True, fancybox=True, shadow=True)
        ax.set_title(
            f"🚨 SAFETY CHECK: Review This Route Before Use! 🚨\n"
            f"Route in {place}\n"
            f"START: ({start_lat:.6f}, {start_lon:.6f})\n"
            f"END: ({end_lat:.6f}, {end_lon:.6f})",
            fontsize=16,
            pad=30,
            color="darkred",
            weight="bold",
        )

        # Add safety text box
        textstr = "⚠️ VERIFY:\n• Safe for pedestrians/bikes\n• No dangerous intersections\n• Appropriate for age/skill"
        props = dict(boxstyle="round", facecolor="yellow", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Route visualization saved to {save_path}")

        if show_immediately:
            # Make the window pop up prominently
            plt.figure(fig.number)
            mng = plt.get_current_fig_manager()
            try:
                # Try to maximize window on different backends
                if hasattr(mng, "window"):
                    if hasattr(mng.window, "wm_state"):
                        mng.window.wm_state("zoomed")  # Windows
                    elif hasattr(mng.window, "showMaximized"):
                        mng.window.showMaximized()  # Qt
            except:
                pass  # Fallback: just show normally

            # Show and wait for user to close the window
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
