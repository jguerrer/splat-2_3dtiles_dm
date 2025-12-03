# splat-3dtiles

## Introduction

splat-3dtiles is a tool for converting Gaussian point clouds to Cesium 3D Tiles format.

## Demo

- Using https://github.com/yangjs6/mapbox-3d-tiles to load large-scale 3DGS. Click the image to jump to Bilibili to watch the related video.

[![](https://i0.hdslb.com/bfs/archive/0b195aebb064cd5b2222faeda00e94308dc4dea6.jpg@672w_378h_1c.webp)](https://www.bilibili.com/video/BV1qsK3z4Eo5/)


## Data Requirements
Only supports .splat data files, only supports Z-up orientation, and must be stored in ENU coordinate system.
Translation, rotation, and scaling operations are not currently supported. If needed, you can use other tools for conversion first.
You can use tools like SuperSplat for conversion: https://superspl.at/editor


## Workflow

1. Read Gaussian point cloud files and split them into tiles
2. Clean data: remove outliers, filter out points with low transparency and excessive scaling, and merge tiles with the same coordinates
3. Build LOD data: recursively generate parent LOD data from tiles
4. Convert to 3D Tiles: generate GLB files and tileset.json file

## Design Notes
1. Since the data can be very large, files are used for data exchange to fully utilize multi-threaded CPU, and tiles are processed in parallel.
2. Initial data may contain multiple tiles. During splitting, even identical tiles are written to different files and merged later.
3. Intermediate files are not deleted to avoid starting from scratch if errors occur. You can continue from intermediate steps by commenting out code.
4. Tested with data over 10GB, but testing is still not exhaustive.
5. The generated 3D Tiles data can be loaded with Cesium, but the results may not be optimal. It's designed to work with another custom renderer: https://github.com/yangjs6/mapbox-3d-tiles.
If you need to load with Cesium, you can refer to that tool.


## Usage

```
python main.py --input ./data/NNU_1/splats --output ./data/NNU_1/3dtiles --enu_origin 118.91083364082562 32.116922266350315 --tile_zoom 20
```

## Reference Run Configuration
    "configurations": [
        {
            "name": "Python Debugger: splat-3dtiles",
            "type": "debugpy",
            "request": "launch",
            "program": "./main.py",
            "console": "integratedTerminal",
            "python": "D:/Python39/python.exe",
            "args": [
                "--input", "./data/NNU_1/splats", 
                "--output", "./data/NNU_1/3dtiles",
                "--enu_origin", "118.91083364082562", "32.116922266350315",
                "--tile_zoom", "20",                
            ],
        }
    ]

## Complete Parameters

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
    
