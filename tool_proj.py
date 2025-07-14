
import os
import struct
from tqdm import tqdm
from pyproj import Proj, Transformer

from common import get_point_num, getPointSize
from mercator import lat_lon_to_mercator, mercatorZfromAltitude


# 定义投影参数
proj_from = Proj(proj='tmerc', ellps='WGS84', lat_0=0, lon_0=120, k_0=1, x_0=103168.7975, y_0=-3554620.906)

# 中心点为经纬度坐标
# [118.90690433781326, 32.11042979092368]
lon, lat = proj_from(0, 0, inverse=True)
print(f"中心点为经纬度坐标 [{lon}, {lat}] ") 

mercator_constant = mercatorZfromAltitude(1, lat)
ref_mercator_x, ref_mercator_y = lat_lon_to_mercator(lat, lon)
    

def convert_to_mercator(input_file: str, output_file: str):

    point_size = getPointSize()
    point_num = get_point_num(input_file)
    point_i = 0
    point_update = 0
    point_num_per_update = 1000

    
    # 初始化进度条
    pbar = tqdm(total=point_num, desc="proj file", position=0)

    write_file = open(output_file, "w+b")
    read_file = open(input_file, "rb")

    while point_i < point_num:
        point_data = read_file.read(point_size)  # 3个 Float32，每个4字节
        if not point_data:
            break

        point_data = bytearray(point_data)

        position_data = point_data[0:12]
        position = struct.unpack('3f', position_data)
        
        # 转换坐标
        x, y, z = position
        # x1, y1 = transformer.transform(x, y)
        # position1 = (x1 - mercator_x, y1 - mercator_y, z)

        lon, lat = proj_from(x, y, inverse=True)
        mercator_x, mercator_y = lat_lon_to_mercator(lat, lon)
        x1 = (mercator_x - ref_mercator_x) / mercator_constant
        y1 = -(mercator_y - ref_mercator_y) / mercator_constant
        position1 = (x1, y1, z)

        point_data[0:12] =struct.pack('3f', *position1)
        write_file.write(point_data)

        point_i += 1

        # 每隔1000个点通知主进程一次
        if point_i % point_num_per_update == 0 or point_i == point_num - 1:
            progress_update = (point_i - point_update)
            pbar.update(progress_update)  # 更新进度条
            point_update = point_i

    pbar.close()

    read_file.close()
    write_file.close()



if __name__ == "__main__":


    input_dir = './data/NNU_2/splats_proj/'
    output_dir = './data/NNU_2/splats/'


    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取所有 Splat 文件
    splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]  

    for splat_file in splat_files:
        input_file = os.path.join(input_dir, splat_file)
        output_file = os.path.join(output_dir, splat_file)
        print(f"正在转换 {input_file}...")
        
        if not os.path.exists(output_file):
            convert_to_mercator(input_file, output_file)

    print("转换完成")
    

    
