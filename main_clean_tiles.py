import os
from multiprocessing import Pool, cpu_count, Manager
import struct
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from common import get_point_num, getPointSize, read_splat_file, write_splat_file

from point import Point
from tile import TileId

point_num_per_update = 1000

def write_splat_to_file(tile_file_path: str, point_size: int, min_alpha: float, max_scale: float, write_file, file_num: int, progress_queue):

    point_i = 0
    point_update = 0
    point_num = get_point_num(tile_file_path)

    with open(tile_file_path, 'rb') as f:
        while point_i < point_num:

            point_data = f.read(point_size)  # 3 Float32 values, 4 bytes each
            if not point_data:
                break
            
            # Notify main process every 1000 points
            if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                progress_update = (point_i - point_update) / point_num / file_num
                progress_queue.put(progress_update)
                point_update = point_i

            point_i += 1

            position = struct.unpack('3f', point_data[0:12])
            scale = struct.unpack('3f', point_data[12:24])
            color = struct.unpack('4B', point_data[24:28])
            rotation = struct.unpack('4B', point_data[28:32])

            if color[3] < min_alpha or max(abs(s) for s in scale) > max_scale:
                continue  # Skip invalid points
            
            write_file.write(point_data)



# Clean a single tile
def clean_tile(tile_id: TileId, tile_files: List[str], output_dir: str,
               min_alpha: float, max_scale: float, flyers_num: float , flyers_dis: float,
               progress_queue):
    """
    Clean a single tile
    """
    output_tile_file_path = tile_id.getFilePath(output_dir, ".splat")
    if os.path.exists(output_tile_file_path):
        # print(f"Tile {tile_id.toString()} already exists, skipping.")
        progress_queue.put(1)
        progress_queue.put(None)  # Use None as the task completion signal
        return


    if flyers_num > 0:
        all_points = []
        point_i = 0
        point_update = 0


        for tile_file_path in tile_files:
            points = read_splat_file(tile_file_path)
            all_points.extend(points)
        all_point_num = len(all_points)

        mask = np.ones(all_point_num, dtype=bool)  # Initialize mask, keep all points
        
        if all_point_num > 10:
            # Extract positions of all points
            positions = np.array([point.position for point in all_points])
            kdtree = KDTree(positions)
            
            # Logic for removing outliers
            k = max(3, min(flyers_num, all_point_num // 100)) 

            # Calculate average distance for each point
            distances, _ = kdtree.query(positions, k=k+1)  # k+1 includes itself
            avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude itself, calculate average distance

            # Calculate threshold
            threshold = np.mean(avg_distances) + flyers_dis * np.std(avg_distances)

            # Create mask to mark which points to keep
            mask = avg_distances < threshold
        
        # Filter invalid pointsr invalid points
        for i in range(all_point_num):
            point = all_points[i]
            if point.color[3] < min_alpha or max(abs(s) for s in point.scale) > max_scale:
                mask[i] = False  # Mark invalid points

            point_i += 1

            # Notify main process every 1000 points
            if point_i % point_num_per_update == 0 or point_i == all_point_num - 1:
                progress_update = (point_i - point_update) / all_point_num
                progress_queue.put(progress_update)
                point_update = point_i

        # Apply masky mask
        result_points = [all_points[i] for i in range(all_point_num) if mask[i]]
        if len(result_points) > 0:
            write_splat_file(output_tile_file_path, result_points)
    else:
        
        point_size = getPointSize()
        file_num = len(tile_files)

        write_file = open(output_tile_file_path, "w+b")
        writable = write_file.writable()
        if not writable:
            print(f"Error: Cannot write to {output_tile_file_path}.")
            progress_queue.put(1)
            progress_queue.put(None)
        for tile_file_path in tile_files:
            write_splat_to_file(tile_file_path, point_size, min_alpha, max_scale, write_file, file_num, progress_queue)



    # Notify main process that task is complete
    progress_queue.put(None)  # Use None as the task completion signal


def main_clean_tiles(input_dir: str, output_dir: str,
                     min_alpha: float, max_scale: float, flyers_num: float , flyers_dis: float):

    """
    Clean tiles using multiprocessing for parallel processing
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read all Splat files
    splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]

    # Parse all tiles from filesles from files
    splat_tiles = defaultdict(list)
    for splat_file in splat_files:
        tile_id = TileId.fromString(splat_file)
        splat_tiles[tile_id].append(os.path.join(input_dir, splat_file))

    # Initialize task queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Initialize progress bar
    total_tasks = len(splat_tiles)
    pbar = tqdm(total=total_tasks, desc="Cleaning tiles", position=0)
    pbar.mininterval = 0.01

    # Use multiprocessing to process each parent tile in parallel
    with Pool(processes=cpu_count() - 1) as pool:
        tasks = []
        for tile_id, tile_files in splat_tiles.items():
            tasks.append(pool.apply_async(clean_tile, (tile_id, tile_files, output_dir, min_alpha, max_scale, flyers_num, flyers_dis, progress_queue)))

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