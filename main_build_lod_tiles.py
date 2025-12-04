import os
import time
import struct
from multiprocessing import Manager, Pool, cpu_count
from typing import Dict, List, Tuple
from collections import defaultdict

from tqdm import tqdm

from common import getPointSize, read_splat_file, write_splat_file

from point import Point
from tile import TileId

import numpy as np
from scipy.spatial import KDTree


point_num_per_update = 1000



def build_lod_tiles_for_parent(parent_tile_id: TileId, children_tile_ids: List[TileId], input_dir: str, output_dir: str, distance_threshold: float, progress_queue):
    """
    Build LOD for a single parent tile
    """
    try:
        parent_tile_file_path = parent_tile_id.getFilePath(output_dir, ".splat")

        parent_points = []
        for child_tile_id in children_tile_ids:
            child_tile_file_path = child_tile_id.getFilePath(input_dir, ".splat")        
            points = read_splat_file(child_tile_file_path)
            parent_points.extend(points)

        lod_points = []
        point_num = len(parent_points)
        if point_num == 0:
            return lod_points

        # Extract positions of all points
        positions = np.array([point.position for point in parent_points])

        # Build KDTree
        kdtree = KDTree(positions)
        visited = np.zeros(point_num, dtype=bool)

        point_update = 0
        for i in range(point_num):
            # Notify main process every 1000 points
            if i % point_num_per_update == 0 or i == point_num - 1:
                progress_update = (i - point_update) / point_num
                progress_queue.put(progress_update)
                point_update = i

            if visited[i]:
                continue

            # Query neighborhood of current point
            indices = kdtree.query_ball_point(positions[i], distance_threshold)

            # Mark these points as visited
            visited[indices] = True

            # Extract points in the cluster points in the cluster
            cluster_points = [parent_points[j] for j in indices]
            weights = np.array([point.color[3] / 255.0 for point in cluster_points])

            # Calculate weighted average position
            weighted_positions = np.average([point.position for point in cluster_points], axis=0, weights=weights)
            # Calculate weighted average color
            weighted_color = np.average([point.color for point in cluster_points], axis=0, weights=weights)
            # Calculate weighted average scale
            # weighted_scale = np.average([point.scale for point in cluster_points], axis=0, weights=weights)
            # Calculate weighted average rotation
            weighted_rotation = np.average([point.rotation for point in cluster_points], axis=0, weights=weights)

            # Calculate distribution range of pointse distribution range of points
            cluster_positions = np.array([point.position for point in cluster_points])

            min_pos = max_pos = np.array(weighted_positions)

            # Calculate boundary for each point
            for point in cluster_points:
                p1 = np.array(point.position) - np.array(point.scale)
                p2 = np.array(point.position) + np.array(point.scale)
                min_pos = np.minimum(min_pos, p1)
                max_pos = np.maximum(max_pos, p2)
                
            weighted_scale = (max_pos - min_pos) / 2


            weighted_color = np.clip(weighted_color, 0, 255)  # Limit range
            weighted_color = np.round(weighted_color).astype(int)  # Round and convert to integer

            weighted_rotation = np.clip(weighted_rotation, 0, 255)  # Limit range
            weighted_rotation = np.round(weighted_rotation).astype(int)  # Round and convert to integer

            lod_points.append(Point(weighted_positions, weighted_color, weighted_scale, weighted_rotation))

        write_splat_file(parent_tile_file_path, lod_points)
        
        # Notify main process that task is complete
        progress_queue.put(None)  # Use None as the task completion signal
    except Exception as e:
        print(f"Error in build_lod_tiles_for_parent: {e}")
        progress_queue.put(None)  # Ensure main process doesn't block


def main_build_lod_tiles(input_dir: str, output_dir: str,
                         enu_origin: Tuple[float, float] = (0.0, 0.0),
                         tile_zoom: int = 20, tile_resolution: float = 0.1):
    """
    Build LOD tiles using multiprocessing for parallel processing
    """

    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    distance_threshold = tile_resolution * (2** (20 - tile_zoom))
    # Read all Splat files
    splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]

    # Parse all tiles from filesles from files
    splat_tiles: List[TileId] = []
    for splat_file in splat_files:
        tile_id = TileId.fromString(splat_file)
        splat_tiles.append(tile_id)

    parent_tiles = defaultdict(list)
    for tile_id in splat_tiles:
        parent_tile_id = tile_id.getParent()
        parent_tiles[parent_tile_id].append(tile_id)

    # Initialize progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Initialize progress bar
    total_tasks = len(parent_tiles)
    pbar = tqdm(total=total_tasks, desc="Building lod", position=0)
    pbar.mininterval = 0.01

    # Use multiprocessing to process each parent tile in parallel
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for parent_tile_id, children_tile_ids in parent_tiles.items():
            tasks.append(pool.apply_async(build_lod_tiles_for_parent, (parent_tile_id, children_tile_ids, input_dir, output_dir, distance_threshold, progress_queue)))

        # Wait for all tasks to complete
        completed_tasks = 0
        while completed_tasks < total_tasks:
            progress_update = progress_queue.get()  # Wait for subprocess to notify progress

            if progress_update is None:
                completed_tasks += 1  # Task completion signal
            else:
                pbar.update(progress_update)  # Update progress bar

        # Wait for all tasks to complete
        for task in tasks:
            task.get()

    # Close progress bar
    pbar.close()