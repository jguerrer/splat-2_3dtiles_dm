

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

if __name__ == "__main__":
    
    output_dir = ''
    min_alpha = 1
    max_scale = 10000
    flyers_num = 25
    flyers_dis = 10
    
    """
    Clean a single tile
    """
    input_tile_file_path = 'E:/Works/CityFun/mapcube/mapcube-demos/car-controller/public/assets/splats/surveyhouse.splat'
    output_tile_file_path = 'E:/Works/CityFun/mapcube/mapcube-demos/car-controller/public/assets/splats/surveyhouse2.splat'



    if flyers_num > 0:
        all_points = []
        point_i = 0
        point_update = 0

        all_points = read_splat_file(input_tile_file_path)

        all_point_num = len(all_points)

        mask = np.ones(all_point_num, dtype=bool)  # Initialize mask, keep all points
        
        if all_point_num > 10:
            # Extract positions of all pointspositions of all points
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

            # # Notify main process every 1000 points
            # if point_i % point_num_per_update == 0 or point_i == all_point_num - 1:
            #     progress_update = (point_i - point_update) / all_point_num
            #     progress_queue.put(progress_update)
            #     point_update = point_i

        # Apply mask
        result_points = [all_points[i] for i in range(all_point_num) if mask[i]]
        if len(result_points) > 0:
            write_splat_file(output_tile_file_path, result_points)
            