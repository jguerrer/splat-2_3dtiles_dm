import struct
from typing import List, Tuple
import numpy as np

# Define point data structure
class Point:
    def __init__(self, position: Tuple[float, float, float], color: Tuple[int, int, int, int],
                 scale: Tuple[float, float, float], rotation: Tuple[int, int, int, int]):
        self.position = position
        self.color = color
        self.scale = scale
        self.rotation = rotation

    def to_bytes(self) -> bytes:
        """Pack point data into binary format"""
        return struct.pack('3f4B3f4B', *self.position, *self.color, *self.scale, *self.rotation)

    @classmethod
    def from_bytes(cls, data: bytes):
        """Parse point from binary data"""
        unpacked = struct.unpack('3f4B3f4B', data)
        position = unpacked[:3]
        color = unpacked[3:7]
        scale = unpacked[7:10]
        rotation = unpacked[10:]
        return cls(position, color, scale, rotation)


def compute_box(points: List[Point]) -> List[float]:    
    positions = np.array([point.position for point in points])
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    center = (min_coords + max_coords) / 2
    half_size = (max_coords - min_coords) / 2
    return [center[0], center[1], center[2], half_size[0], 0, 0, 0, half_size[1], 0, 0, 0, half_size[2]]


def merge_box(box_list: List[List[float]]) -> List[float]:
    """
    Merge multiple bounding boxes
    :param box_list: A list containing multiple bounding boxes, each bounding box is a list of length 12
    :return: Merged bounding box, also a list of length 12
    """
    if not box_list:
        raise ValueError("box_list cannot be empty")

    # Extract center points and half sizes of all bounding boxespoints and half sizes of all bounding boxes
    centers = np.array([box[:3] for box in box_list])
    half_sizes = np.array([box[3::4] for box in box_list])

    # Calculate minimum and maximum coordinates of all bounding boxes
    min_coords = np.min(centers - half_sizes, axis=0)
    max_coords = np.max(centers + half_sizes, axis=0)

    # Calculate center point and half size of merged bounding box
    merged_center = (min_coords + max_coords) / 2
    merged_half_size = (max_coords - min_coords) / 2

    # Construct merged bounding box
    merged_box = list(merged_center) + [merged_half_size[0], 0, 0, 0, merged_half_size[1], 0, 0, 0, merged_half_size[2]]
    return merged_box