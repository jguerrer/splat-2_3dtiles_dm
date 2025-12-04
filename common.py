
__version__ = '0.1'
import os
import struct
from typing import List

from point import Point


point_size = 3*4 + 3*4 + 4*1 + 4*1  # 3 float32 + 3 float32 + 4 uint8 + 4 uint8 = 32 bytes

def getVersion():
    return __version__

def getPointSize():
    return point_size


def get_point_num(file_path: str) -> int:
    stats = os.stat(file_path)    
    point_size = getPointSize()
    point_num = stats.st_size // point_size
    return point_num

# Read data
def read_splat_file(file_path: str) -> List[Point]:
    """
    Read binary format Splat file
    :param file_path: Splat file path
    :return: List of Point objects containing position, scale, color, rotation data
    """
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
            # rotation = (rotation[1], rotation[2], rotation[3], rotation[0])

            points.append(Point(position, color, scale, rotation))
    return points

# Write datae data
def write_splat_file(file_path: str, points: List[Point]):
    """
    Write list of Point objects to binary format Splat file
    :param file_path: Splat file path
    :param points: List of Point objects containing position, scale, color, rotation data
    """
    with open(file_path, 'wb') as f:
        for point in points:
            # Write position (3 Float32)
            f.write(struct.pack('3f', *point.position))
            # Write scale (3 Float32)
            f.write(struct.pack('3f', *point.scale))
            # Write color (4 Bytes)
            f.write(struct.pack('4B', *point.color))
            # Write rotation (4 Bytes), adjust quaternion order (w, x, y, z) -> (x, y, z, w)
            # rotation = (point.rotation[1], point.rotation[2], point.rotation[3], point.rotation[0])
            f.write(struct.pack('4B', *point.rotation))