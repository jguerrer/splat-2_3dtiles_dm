"""
Convert 3D Gaussian Splatting point cloud to Cesium 3D Tiles format.
The glTF files include the KHR_gaussian_splatting extension.
 
Reference
https://github.com/CesiumGS/glTF/tree/proposal-KHR_gaussian_splatting/extensions/2.0/Khronos/KHR_gaussian_splatting

Author: Yang Jianshun 20250528

"""

import argparse
import math
import os
import sys
import time
from mercator import geodetic_to_ecef_transformation
import numpy as np

from point import Point
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
import base64
import struct
import json
from pyproj import Transformer
from typing import List, Dict, Tuple
import argparse

from tile import Tile
from tile_manager import TileManager


__version__ = '0.1'


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
    # normalized_rotations = rotations / 255.0
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
    print(f"glTF file saved to: {output_path}")


# Read data data
def read_splat_file(file_path: str) -> List[Point]:
    """
    Read binary format Splat file
    :param file_path: Splat file path
    :return: List of Point objects containing position, scale, color, rotation data
    """
    
    stats = os.stat(file_path)
    
    point_size = 3*4 + 3*4 + 4*1 + 4*1  # 3 float32 + 3 float32 + 4 uint8 + 4 uint8 = 32 bytes

    point_num = stats.st_size // point_size
    point_i = 0

    points = []
    with open(file_path, 'rb') as f:
        while True:
            position_data = f.read(3 * 4)  # 3 Float32 values, 4 bytes each
            if not position_data:
                break
            position = struct.unpack('3f', position_data)
            scale = struct.unpack('3f', f.read(3 * 4))
            color = struct.unpack('4B', f.read(4 * 1))
            rotation = struct.unpack('4B', f.read(4 * 1))
            # Adjust quaternion order (x, y, z, w) -> (w, x, y, z)
            rotation = (rotation[1], rotation[2], rotation[3], rotation[0])
            points.append(Point(position, color, scale, rotation))

            # progress = (point_i / point_num) * 100
            # finsh = "▓" * (int)(progress)
            # need_do = "-" * (int)(100 - progress)
            # print("\r{:^3.0f}%[{}->{}]".format(progress, finsh, need_do), end="")
            # point_i += 1
    return points


# Calculate box range




# NumpyEncoder for serializing NumPy arrays


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_tile_gltf_filename(tile: Tile):
    tileId = tile.getTileId().toString()
    return f"{tileId}.gltf"


def generate_tileset_json(output_dir: str, tile_manager: TileManager, geometric_error: int = 100):
    def build_tile_structure(tile: Tile, current_geometric_error: int) -> Dict:
        # If point count is 0, return None
        if len(tile.getPoints()) == 0:
            return {}

        bounding_volume = {"box": tile.getBounds()}
        
        tile_gltf = get_tile_gltf_filename(tile)

        content = {"uri": tile_gltf} 
        
        tile_structure = {
            "boundingVolume": bounding_volume,
            "geometricError": current_geometric_error,
            "refine": "REPLACE",
            "content": content
        }
        return tile_structure
    
    def build_root(tile_manager:TileManager, current_geometric_error: int):
        tiles = tile_manager.getTiles()

        bounding_volume = {"box": tile_manager.getBounds()}
        transform = tile_manager.getTransform()

        children = [build_tile_structure(tile, current_geometric_error / 2)
                    for tile in tiles] if tiles else []
        tile_structure = {
            "boundingVolume": bounding_volume,
            "transform": transform,
            "geometricError": current_geometric_error,
            "refine": "REPLACE",
        }
        if children:
            tile_structure["children"] = children
        return tile_structure

    tileset = {
        "asset": {"version": "1.1", "gltfUpAxis": "Z"},
        "geometricError": geometric_error,
        "root": build_root(tile_manager, geometric_error)
    }

    with open(f"{output_dir}/tileset.json", "w") as f:
        json.dump(tileset, f, cls=NumpyEncoder, indent=4)


def splat_to_3dtiles_file(input_file: str, output_dir: str, enu_origin: Tuple[float, float],
                       tile_zoom: float = 20,
                       min_alpha: float = 1.0, max_scale: float = 10000) -> None:
    """
    Convert Splat point cloud to Cesium 3D Tiles format
    :param input_file: Input Splat point cloud file path
    :param output_dir: Output 3D Tiles folder
    :param tile_center: Center coordinates of point cloud (x, y, z)
    :param min_alpha: Minimum alpha threshold, Gaussian points below this will be filtered
    :param max_scale: Maximum scale threshold, Gaussian points above this will be filtered
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    time1 = time.perf_counter()     
    
    # Read Splat file
    points = read_splat_file(input_file)

    time2 = time.perf_counter() 
    print(f"\nRead {len(points)} points, took {(time2- time1):.2f} seconds")

    # Filter points
    filtered_points = [
        point for point in points
        if point.scale[0] <= max_scale and point.scale[1] <= max_scale and point.scale[2] <= max_scale
        and point.color[3] >= min_alpha
    ]
    
    time3 = time.perf_counter() 
    print(f"After filtering, {len(filtered_points)} points remain {(time3 - time2):.2f} seconds")

    # If no points, return directly
    if not filtered_points:
        print("No points meet the criteria, conversion ended.")
        return
    
    tile_manager = TileManager(enu_origin, tile_zoom) 
    # Add points to tile manager
    
    tile_manager.setPoints(filtered_points)
    tile_manager.buildLOD()
    # Get all tiles
    tiles = tile_manager.getTiles()

    time4 = time.perf_counter() 
    print(f"Generated {len(tiles)} tiles in {(time4 - time3):.2f} seconds")

    for tile in tiles:
        points = tile.getPoints()
        tile_gltf = get_tile_gltf_filename(tile)
        output_file = os.path.join(output_dir, tile_gltf)
        # If tile has no points, skip
        if not points:
            continue
        # Convert points to glTF format
        splat_to_gltf_with_gaussian_extension(points, output_file)
    
    generate_tileset_json(output_dir, tile_manager)
    
    time5 = time.perf_counter() 
    print(f"Generated {len(tiles)} gltf files in {(time5 - time4):.2f} seconds")


# Convert point cloud data to Cesium 3D Tiles format Cesium 3D Tiles format
def splat_to_3dtiles_main(input_dir: str, output_dir: str, 
                          enu_origin: Tuple[float, float] = (0.0, 0.0), tile_zoom: float = 20,
                        tile_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                        min_alpha: float = 1.0, max_scale: float = 10000,
                        tile_size: float = 100, min_point_num: int = 10000):
        """
        Convert Splat point cloud to Cesium 3D Tiles format
        :param input_dir: Input Gaussian point cloud folder
        :param output_dir: Output folder to save 3D Tiles
        :param enu_origin: Origin longitude and latitude of ENU coordinate system (lon, lat)
        :param tile_center: Center coordinates of point cloud (x, y, z)
        :param min_alpha: Minimum alpha threshold
        :param max_scale: Maximum scale threshold
        :param tile_size: Minimum tile size
        :param min_point_num: Minimum tile point count
        """
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read all Splat files
        splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]        
        for splat_file in splat_files:
            file_path = os.path.join(input_dir, splat_file)
            splat_to_3dtiles_file(
                input_file=file_path,
                output_dir=output_dir,
                enu_origin=enu_origin,
                tile_zoom=tile_zoom,
                min_alpha=min_alpha,
                max_scale=max_scale
            )

# Main function
if __name__ == "__main__":

    print(f"splat-3dtiles: {__version__}")
    
    # Parse command line arguments
    
    parser = argparse.ArgumentParser(description="Convert 3D Gaussian Splatting point cloud to Cesium 3D Tiles format")
    parser.add_argument("--input", "-i", required=True, help="Input Gaussian point cloud folder.")
    parser.add_argument("--output", "-o", required=True, help="Output folder to save 3D Tiles.")
    parser.add_argument("--enu_origin", nargs=2, type=float, metavar=('lon', 'lat'), help="Specify the origin longitude and latitude (lon, lat) of the ENU coordinate system. Default is (0.0, 0.0).")
    parser.add_argument("--tile_zoom", type=float, default=20, help="Tile zoom level. Default is 20.")
    parser.add_argument("--tile_center", nargs=3, type=float, metavar=('x', 'y', 'z'), help="Specify the center coordinates of the point cloud (x, y, z). Default is (0.0, 0.0, 0.0).")
    parser.add_argument("--tile_size", type=float, default=100, help="Minimum tile size, tiles smaller than this value will not be further divided. Default is 100 meters.")
    parser.add_argument("--min_alpha", type=float, default=1.0, help="Minimum alpha threshold. Gaussian points below this threshold will be filtered. Default is 1.0.")
    parser.add_argument("--max_scale", type=float, default=10000, help="Maximum scale threshold. Gaussian points above this threshold will be filtered. Default is 10000.")
    parser.add_argument("--min_point_num", type=int, default=10000, help="Minimum tile point count, tiles with fewer points will not be further divided. Default is 10000 points.")
    args = parser.parse_args()



    splat_to_3dtiles_main(
        input_dir=args.input,
        output_dir=args.output,
        enu_origin=(args.enu_origin[0], args.enu_origin[1]) if args.enu_origin else (0.0, 0.0),
        tile_zoom=args.tile_zoom if args.tile_zoom else 20,
        tile_center=(args.tile_center[0], args.tile_center[1], args.tile_center[2]) if args.tile_center else (0.0, 0.0, 0.0),
        min_alpha=args.min_alpha if args.min_alpha else 1.0,
        max_scale=args.max_scale if args.max_scale else 10000,
        tile_size=args.tile_size if args.tile_size else 100,
        min_point_num=args.min_point_num if args.min_point_num else 10000,
    )

