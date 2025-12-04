"""
Convert 3D Gaussian Splatting point cloud to Cesium 3D Tiles format.
The glTF files include the KHR_gaussian_splatting extension.
 
Reference
https://github.com/CesiumGS/glTF/tree/proposal-KHR_gaussian_splatting/extensions/2.0/Khronos/KHR_gaussian_splatting

Author: Yang Jianshun 20250528

"""

import argparse
from multiprocessing import freeze_support
import os
from common import getVersion

from main_convert_to_3dtiles import main_convert_to_3dtiles
from main_convert_to_gltf import main_convert_to_gltf
from main_split_to_tiles import main_split_to_tiles
from main_clean_tiles import main_clean_tiles
from main_build_lod_tiles import main_build_lod_tiles


# Main function

# general  workflow is:
# 1 ) Split original splat files into tiles
# 2 ) Clean tiles to remove outliers and invalid points 
# 3 ) Build LOD tiles from cleaned tiles
# 4 ) Convert built LOD tiles to 3D Tiles format

# considerations involve providing ENU coordinates
if __name__ == "__main__":
    freeze_support()
    
    __version__ = getVersion()
    print(f"splat-3dtiles: {__version__}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert 3D Gaussian Splatting point cloud to Cesium 3D Tiles format")
    parser.add_argument("--input", "-i", required=True, help="Input Gaussian point cloud folder.")
    parser.add_argument("--output", "-o", required=True, help="Output folder to save 3D Tiles.")
    parser.add_argument("--enu_origin", nargs=2, type=float, metavar=('lon', 'lat'), help="Specify the origin longitude and latitude (lon, lat) of the ENU coordinate system. Default is (0.0, 0.0).")
    parser.add_argument("--tile_zoom", type=int, default=20, help="Tile zoom level. Default is 20.")
    parser.add_argument("--tile_resolution", type=float, default=0.1, help="Parameter for generating LOD, representing the precision at zoom level 20. Default is 0.1 meters.")
    parser.add_argument("--tile_error", type=float, default=1, help="geometric_error parameter for generating tileset.json, representing the error at zoom level 20. Default is 1 meter.")


    parser.add_argument("--min_alpha", type=float, default=1.0, help="Minimum alpha threshold. Gaussian points below this threshold will be filtered. Default is 1.0.")
    parser.add_argument("--max_scale", type=float, default=10000, help="Maximum scale threshold. Gaussian points above this threshold will be filtered. Default is 10000.")
    parser.add_argument("--flyers_num", type=int, default=25, help="Number of nearest neighbors for removing outliers. Default is 25.")
    parser.add_argument("--flyers_dis", type=float, default=10, help="Distance factor for removing outliers. Smaller values remove more points. Default is 10.")
    
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    enu_origin = (args.enu_origin[0], args.enu_origin[1]) if args.enu_origin else (0.0, 0.0)
    tile_zoom = args.tile_zoom    
    tile_resolution = args.tile_resolution
    tile_error = args.tile_error
    
    min_alpha = args.min_alpha
    max_scale = args.max_scale
    flyers_num = args.flyers_num
    flyers_dis = args.flyers_dis


    split_output_dir = os.path.join(output_dir, f"split")
    build_output_dir = os.path.join(output_dir, f"build")
    result_output_dir = os.path.join(output_dir, f"result")

    clean_output_dir = os.path.join(build_output_dir, f"{tile_zoom}")

    print(f"----main_split_to_tiles start:[{tile_zoom}][{input_dir}][{split_output_dir}]")
    main_split_to_tiles(input_dir, split_output_dir, enu_origin, tile_zoom)

    print(f"----main_clean_tiles start:[{tile_zoom}][{split_output_dir}][{clean_output_dir}]")
    main_clean_tiles(split_output_dir, clean_output_dir, min_alpha, max_scale, flyers_num, flyers_dis)


    lod_zoom = tile_zoom - 1
    lod_input_dir = clean_output_dir
    while lod_zoom > tile_zoom - 6:
        lod_output_dir = os.path.join(build_output_dir, f"{lod_zoom}")

        print(f"----main_build_lod_tiles start:[{lod_zoom}][{lod_input_dir}][{lod_output_dir}]")
        main_build_lod_tiles(lod_input_dir, lod_output_dir, enu_origin, lod_zoom, tile_resolution)

        lod_input_dir = lod_output_dir
        lod_zoom -= 1
    
    #fails
    main_convert_to_3dtiles(build_output_dir, result_output_dir, enu_origin, tile_zoom, tile_error)
    