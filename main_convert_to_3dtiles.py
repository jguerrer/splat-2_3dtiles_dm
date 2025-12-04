from collections import defaultdict
import json
import os
from typing import List, Tuple, Dict

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
from point import Point, compute_box, merge_box
from tile import TileId
from tile_manager import TileManager

from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor

import numpy as np

from mercator import geodetic_to_ecef_transformation
from tile import TileId



class TileNode:
    def __init__(self, tile_id: TileId):
        self.tile_id = tile_id  # Tile ID
        self.geometric_error = 0
        self.bounds = []
        self.children = list()
        self.splat_file = ''
        self.gltf_file = ''
        self.parent = None


def build_tile_tree(input_dir: str, tile_error: float = 1) -> (List[TileNode], Dict[TileId, TileNode]):
    # Properly initialize defaultdict
    tile_node_dict: Dict[TileId, TileNode] = defaultdict(TileNode)

    input_sub_dirs = [dir for dir in os.listdir(input_dir)]
    for sub_dir in input_sub_dirs:
        input_sub_dir = os.path.join(input_dir, sub_dir)

        # Read all Splat files
        splat_files = [f for f in os.listdir(input_sub_dir) if f.endswith('.splat')]  
        
        for splat_file in splat_files:
            tile_id = TileId.fromString(splat_file)
            tile_node = TileNode(tile_id)
            tile_node.geometric_error = tile_error * (2** (20 - tile_id.z))
            tile_node.splat_file = os.path.join(sub_dir, splat_file)
            tile_node.splat_file = tile_node.splat_file.replace("\\", "/")
            gltf_file = splat_file.replace('.splat', '.glb')
            tile_node.gltf_file = os.path.join(sub_dir, gltf_file)
            tile_node.gltf_file = tile_node.gltf_file.replace("\\", "/")
            tile_node_dict[tile_id] = tile_node
        
    for tile_id, tile_node in tile_node_dict.items():
        parent_tile_id = tile_id.getParent()
        parent_tile_node = tile_node_dict.get(parent_tile_id)
        if parent_tile_node:
            parent_tile_node.children.append(tile_node)
            tile_node.parent = parent_tile_node

    root_tile_nodes = []
    for tile_id, tile_node in tile_node_dict.items():
        if tile_node.parent is None:
            root_tile_nodes.append(tile_node)

    return root_tile_nodes, tile_node_dict

# NumpyEncoder for serializing NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        # elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_tileset_json(root_tile_nodes: List[TileNode], output_dir: str, enu_origin: Tuple[float, float]):
    def build_tile_structure(tile_node: TileNode) -> Dict:

        bounding_volume = {"box": tile_node.bounds}
        content = {"uri": tile_node.gltf_file} 
        geometric_error = tile_node.geometric_error
        
        children = [build_tile_structure(child_tile_node)
                    for child_tile_node in tile_node.children] if tile_node.children else []
        
        tile_structure = {
            "boundingVolume": bounding_volume,
            "geometricError": geometric_error,
            "refine": "REPLACE",
            "content": content,
        }
        
        if children:
            tile_structure["children"] = children

        return tile_structure
    
    def build_root(root_tile_nodes:List[TileNode], enu_origin: Tuple[float, float], geometric_error: float):

        box_list = [tile_node.bounds for tile_node in root_tile_nodes] if root_tile_nodes else []
        
        bounding_volume = {"box": merge_box(box_list)}

        # Convert longitude and latitude to ECEF transformation matrix
        transform = geodetic_to_ecef_transformation(enu_origin[0], enu_origin[1])

        children = [build_tile_structure(tile_node)
                    for tile_node in root_tile_nodes] if root_tile_nodes else []
        
        tile_structure = {
            "boundingVolume": bounding_volume,
            "transform": transform,
            "geometricError": geometric_error,
            "refine": "REPLACE",
        }
        if children:
            tile_structure["children"] = children
        return tile_structure

    geometric_error_list = [tile_node.geometric_error for tile_node in root_tile_nodes] if root_tile_nodes else []
    geometric_error = np.max(geometric_error_list)

    tileset = {
        "asset": {"version": "1.1", "gltfUpAxis": "Z"},
        "geometricError": geometric_error,
        "root": build_root(root_tile_nodes, enu_origin, geometric_error)
    }

    with open(f"{output_dir}/tileset.json", "w") as f:
        json.dump(tileset, f, cls=NumpyEncoder, indent=4)


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

    def create_accessor(buffer_view: int, component_type: int, normalized: bool, count: int, type: str, max: List[float] = None, min: List[float] = None) -> Accessor:
        return Accessor(bufferView=buffer_view, componentType=component_type, normalized=normalized, count=count, type=type, max=max, min=min)

    buffer_views = [
        create_buffer_view(0, positions_binary),
        create_buffer_view(len(positions_binary), colors_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary), rotations_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary) + len(rotations_binary), scales_binary)
    ]
    accessors = [
        create_accessor(0, 5126, False, len(positions), "VEC3", positions.max(
            axis=0).tolist(), positions.min(axis=0).tolist()),
        create_accessor(1, 5121, True, len(colors), "VEC4"),
        create_accessor(2, 5126, False, len(normalized_rotations), "VEC4"),
        create_accessor(3, 5126, False, len(scales), "VEC3")
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


def convert_to_gltf(tile_node: TileNode, input_file: str, output_file: str, shared_tile_node_dict) -> None:
    """
    Convert single Gaussian splatting data file to glTF file and update bounds information
    """
    points = read_splat_file(input_file)
    if(len(points) == 0):
        return

    # Calculate bounds
    tile_node.bounds = compute_box(points)

    # Update bounds information in shared dictionary
    shared_tile_node_dict[tile_node.tile_id] = tile_node.bounds

    # Convert to glTF file
    splat_to_gltf_with_gaussian_extension(points, output_file)

    # Notify main process that task is complete
    # progress_queue.put(None)  # Can keep this line if progress bar is needed


def convert_to_gltf_tiles(tile_node_dict: Dict[TileId, TileNode], input_dir: str, output_dir: str):
    """
    Convert splat to gltf using multiprocessing for parallel processing
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize shared dictionary
    manager = Manager()
    shared_tile_node_dict = manager.dict()

    # Initialize progress bar
    file_num = len(tile_node_dict)
    pbar = tqdm(total=file_num, desc="Convert gltf", position=0)
    pbar.mininterval = 0.01

    # Use multiprocessing for parallel processing
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for tile_id, tile_node in tile_node_dict.items():
            input_file_path = os.path.join(input_dir, tile_node.splat_file)
            output_file_path = os.path.join(output_dir, tile_node.gltf_file)
            if not os.path.exists(os.path.dirname(output_file_path)):
                os.makedirs(os.path.dirname(output_file_path))
            tasks.append(pool.apply_async(convert_to_gltf, (tile_node, input_file_path, output_file_path, shared_tile_node_dict)))

        # Wait for all tasks to complete
        for task in tasks:
            task.get()
            pbar.update(1)  # Update progress bar

    # Close progress bar
    pbar.close()

    # Update tile_node_dict in main process
    for tile_id, bounds in shared_tile_node_dict.items():
        tile_node_dict[tile_id].bounds = bounds


def main_convert_to_3dtiles(input_dir: str, output_dir: str, 
                            enu_origin: Tuple[float, float] = (0.0, 0.0),
                            tile_zoom: int = 20, tile_error: float = 1.0):
    """
    Main function, convert splat files to 3D Tiles
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    root_tile_nodes, tile_node_dict = build_tile_tree(input_dir, tile_error)

    convert_to_gltf_tiles(tile_node_dict, input_dir, output_dir)

    generate_tileset_json(root_tile_nodes, output_dir, enu_origin)

        