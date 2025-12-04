import base64
from multiprocessing import Pool, cpu_count, Manager
import os
import time
import struct
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from common import get_point_num, getPointSize, read_splat_file
from point import Point
from tile import TileId
from tile_manager import TileManager

from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor


# Convert Splat data to glTF file
def splat_to_gltf_with_gaussian_extension(points: List[Point], output_path: str):
    """
    Convert Splat data to glTF file with KHR_gaussian_splatting extension support
    :param points: List of Point objects
    :param output_path: Output glTF file path
    """
    # Extract data
    positions = np.array(
        [point.position for point in points], dtype=np.float32)
    colors = np.array([point.color for point in points], dtype=np.uint8)
    scales = np.array([point.scale for point in points], dtype=np.float32)
    rotations = np.array([point.rotation for point in points], dtype=np.uint8)

    # Adjust quaternion order (w, x, y, z) -> (x, y, z, w)
    rotations = rotations[:, [1, 2, 3, 0]]
    normalized_rotations = ((rotations-128.0)/128.0).astype(np.float32)

    # Create GLTF object
    gltf = GLTF2()
    gltf.extensionsUsed = ["KHR_gaussian_splatting"]

    # Create Buffer
    buffer = Buffer()
    gltf.buffers.append(buffer)

    # Convert data to binary
    positions_binary = positions.tobytes()
    colors_binary = colors.tobytes()
    scales_binary = scales.tobytes()
    rotations_binary = normalized_rotations.tobytes()

    # Create BufferView and Accessor
    def create_buffer_view(byte_offset: int, data: bytes, target: int = 34962) -> BufferView:
        return BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data), target=target)

    def create_accessor(buffer_view: int, component_type: int, count: int, type: str, max: List[float] = None, min: List[float] = None) -> Accessor:
        return Accessor(bufferView=buffer_view, componentType=component_type, count=count, type=type, max=max, min=min)

    buffer_views = [
        create_buffer_view(0, positions_binary),
        create_buffer_view(len(positions_binary), colors_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary), rotations_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary) + len(rotations_binary), scales_binary)
    ]
    accessors = [
        create_accessor(0, 5126, len(positions), "VEC3", positions.max(
            axis=0).tolist(), positions.min(axis=0).tolist()),
        create_accessor(1, 5121, len(colors), "VEC4"),
        create_accessor(2, 5126, len(normalized_rotations), "VEC4"),
        create_accessor(3, 5126, len(scales), "VEC3")
    ]
    gltf.bufferViews.extend(buffer_views)
    gltf.accessors.extend(accessors)

    # Create Mesh and Primitive
    primitive = Primitive(
        attributes={"POSITION": 0, "COLOR_0": 1, "_ROTATION": 2, "_SCALE": 3},
        mode=0,
        extensions={"KHR_gaussian_splatting": {
            "positions": 0, "colors": 1, "scales": 2, "rotations": 3}}
    )
    mesh = Mesh(primitives=[primitive])
    gltf.meshes.append(mesh)

    # Create Node and Scene
    node = Node(mesh=0)
    gltf.nodes.append(node)
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    # Write binary data to Buffer
    gltf.buffers[0].uri = "data:application/octet-stream;base64," + base64.b64encode(
        positions_binary + colors_binary + rotations_binary + scales_binary).decode("utf-8")
    
    gltf.save(output_path)


# Convert single Gaussian splatting data file to tiles
def convert_to_gltf(input_file: str, output_file: str, progress_queue) -> None:

    points = read_splat_file(input_file)
    splat_to_gltf_with_gaussian_extension(points, output_file)
    
    # Notify main process that task is complete
    progress_queue.put(None)  # Use None as the task completion signal


# Convert Gaussian splatting data to tilesussian splatting data to tiles
def main_convert_to_gltf(input_dir: str, output_dir: str, 
                        enu_origin: Tuple[float, float] = (0.0, 0.0),
                        tile_zoom: int = 20):

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        
    input_sub_dirs = [dir for dir in os.listdir(input_dir)]
    for sub_dir in input_sub_dirs:
        input_sub_dir = os.path.join(input_dir, sub_dir)
        output_sub_dir = os.path.join(output_dir, sub_dir)
        
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        # Read all Splat files
        splat_files = [f for f in os.listdir(input_sub_dir) if f.endswith('.splat')]  
        file_num = len(splat_files)

        # Initialize progress queue
        manager = Manager()
        progress_queue = manager.Queue()

        # Initialize progress bar
        pbar = tqdm(total=file_num, desc="Convert gltf", position=0)
        pbar.mininterval = 0.01

        # Use multiprocessing for parallel processing
        with Pool(processes=cpu_count()) as pool:
            tasks = []
            for splat_file in splat_files:
                input_file_path = os.path.join(input_sub_dir, splat_file)

                gltf_file = splat_file.replace('.splat', '.glb')
                output_file_path = os.path.join(output_sub_dir, gltf_file)
                

                tasks.append(pool.apply_async(convert_to_gltf, (input_file_path, output_file_path, progress_queue)))

            # Wait for all tasks to complete
            completed_tasks = 0
            while completed_tasks < file_num:
                progress_update = progress_queue.get()  # Wait for subprocess to notify progress

                if progress_update is None:
                    completed_tasks += 1  # Task completion signal
                    
                pbar.update(1)  # Update progress bar

            # Wait for all tasks to complete
            for task in tasks:
                task.get()

        # Close progress bar
        pbar.close()
