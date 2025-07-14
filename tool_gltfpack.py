import os
import subprocess
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

def compress_gltf_file(input_file: str, output_file: str, progress_queue) -> None:
    """
    使用 gltfpack 压缩单个 glb 文件
    :param file_path: glb 文件的完整路径
    """

    # 通知主进程任务完成
    progress_queue.put(None)  # 使用 None 作为任务完成的信号
    
    current_directory = os.getcwd()
    command = current_directory + "/bin/gltfpack.exe -i " + input_file + " -o " + output_file + " -cc -vpf "

    os.system(command)



def collect_gltf_files(root_dir: str) -> list:
    """
    收集指定目录及其子目录下的所有 glb 文件路径
    :param root_dir: 要遍历的根目录
    :return: 包含所有 glb 文件路径的列表
    """
    gltf_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.glb'):
                gltf_files.append(os.path.join(subdir, file))
    return gltf_files

def compress_gltf_files(input_dir: str, output_dir: str) -> None:
    """
    使用多进程并行压缩 glb 文件
    :param root_dir: 要遍历的根目录
    """
    gltf_files = collect_gltf_files(input_dir)

    if not gltf_files:
        print("未找到 glb 文件")
        return
    
    file_num = len(gltf_files)

    # 获取 CPU 核心数
    num_processes = cpu_count()
    print(f'开始使用 {num_processes} 个进程进行并行压缩...')

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化进度队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化进度条
    pbar = tqdm(total=file_num, desc="Convert gltf", position=0)
    pbar.mininterval = 0.01


    # 使用多进程并行处理切块
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for input_file in gltf_files:
            output_file = input_file.replace(input_dir, output_dir)
            
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            tasks.append(pool.apply_async(compress_gltf_file, (input_file, output_file, progress_queue)))

        # 等待所有任务完成
        completed_tasks = 0
        while completed_tasks < file_num:
            progress_update = progress_queue.get()  # 等待子进程通知进度

            if progress_update is None:
                completed_tasks += 1  # 任务完成信号
                
            pbar.update(1)  # 更新进度条

            # 等待所有任务完成
            for task in tasks:
                task.get()


if __name__ == "__main__":
    input_dir = os.path.abspath('./data/NNU_2/3dtiles/result/')
    output_dir = os.path.abspath('./data/NNU_2/3dtiles/result_opt/')
    compress_gltf_files(input_dir, output_dir)
    print("所有 glb 文件的并行压缩完成！")