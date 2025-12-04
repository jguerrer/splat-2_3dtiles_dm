from multiprocessing import Pool, cpu_count, Manager
import os
import time
import struct
from typing import Dict, Tuple
from collections import defaultdict

from tqdm import tqdm
from common import get_point_num, getPointSize
from tile import TileId
from tile_manager import TileManager

point_num_per_update = 1000

# Write a single Gaussian point's data to a tile file
def write_point_to_tile_file(point_data, tile_id: TileId, tile_files: Dict, output_dir: str, input_id: int = 0):
    write_file = tile_files.get(tile_id)
    if not write_file:
        ext = f".{input_id}.splat"
        output_file = tile_id.getFilePath(output_dir, ext)
        write_file = open(output_file, "w+b")
        tile_files[tile_id] = write_file

    write_file.write(point_data)


# Split a single Gaussian splatting data file into tiles
def split_to_tiles_file(input_file: str, output_dir: str, 
                        tile_manager: TileManager, progress_queue) -> None:

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_id = hash(input_file)
     
    point_size = getPointSize()
    point_num = get_point_num(input_file)
    point_i = 0
    point_update = 0

    tile_files = defaultdict(list)

    with open(input_file, 'rb') as f:
        while point_i < point_num:
            point_data = f.read(point_size)  # 3 Float32 values, 4 bytes each
            if not point_data:
                break

            position_data = point_data[0:12]
            position = struct.unpack('3f', position_data)
            tile_id = tile_manager.getTileId(position)
            write_point_to_tile_file(point_data, tile_id, tile_files, output_dir, input_id)

            point_i += 1

            # Notify main process every 1000 points
            if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                progress_update = (point_i - point_update) / point_num
                progress_queue.put(progress_update)
                point_update = point_i

    for tile_file in tile_files.values():
        tile_file.close()

    # Notify main process that task is complete
    progress_queue.put(None)  # Use None as the task completion signal


# Split Gaussian splatting data into tiles
def main_split_to_tiles(input_dir: str, output_dir: str, 
                        enu_origin: Tuple[float, float] = (0.0, 0.0),
                        tile_zoom: int = 20):

    # Delete all files and subdirectories in the directory
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(f"Directory {output_dir} has been cleared.")
    else:
        print(f"Directory {output_dir} does not exist.")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read all Splat files
    splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]  
    file_num = len(splat_files)

    # Initialize progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Initialize tile manager
    tile_manager = TileManager(enu_origin, tile_zoom)


    # Initialize progress barlize progress bar
    pbar = tqdm(total=file_num, desc="Splitting files", position=0)
    pbar.mininterval = 0.01

    # Use multiprocessing for parallel tile splitting
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for splat_file in splat_files:
            file_path = os.path.join(input_dir, splat_file)
            tasks.append(pool.apply_async(split_to_tiles_file, (file_path, output_dir, tile_manager, progress_queue)))

        # Wait for all tasks to complete
        completed_tasks = 0
        while completed_tasks < file_num:
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
