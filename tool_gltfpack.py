import os
import subprocess
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm

def compress_gltf_file(input_file: str, output_file: str, progress_queue) -> None:
    """
    Compress a single glb file using gltfpack
    :param file_path: Full path to the glb file
    """

    # Notify main process that task is complete
    progress_queue.put(None)  # Use None as the task completion signal
    
    current_directory = os.getcwd()
    command = current_directory + "/bin/gltfpack.exe -i " + input_file + " -o " + output_file + " -cc -vpf "

    os.system(command)



def collect_gltf_files(root_dir: str) -> list:
    """
    Collect all glb file paths in the specified directory and its subdirectories
    :param root_dir: Root directory to traverse
    :return: List containing all glb file paths
    """
    gltf_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.glb'):
                gltf_files.append(os.path.join(subdir, file))
    return gltf_files

def compress_gltf_files(input_dir: str, output_dir: str) -> None:
    """
    Compress glb files in parallel using multiprocessing
    :param root_dir: Root directory to traverse
    """
    gltf_files = collect_gltf_files(input_dir)

    if not gltf_files:
        print("No glb files found")
        return
    
    file_num = len(gltf_files)

    # Get CPU core count
    num_processes = cpu_count()
    print(f'Starting parallel compression with {num_processes} processes...')

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Initialize progress barlize progress bar
    pbar = tqdm(total=file_num, desc="Convert gltf", position=0)
    pbar.mininterval = 0.01


    # Use multiprocessing for parallel processing
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for input_file in gltf_files:
            output_file = input_file.replace(input_dir, output_dir)
            
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            tasks.append(pool.apply_async(compress_gltf_file, (input_file, output_file, progress_queue)))

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


if __name__ == "__main__":
    input_dir = os.path.abspath('./data/NNU_2/3dtiles/result/')
    output_dir = os.path.abspath('./data/NNU_2/3dtiles/result_opt/')
    compress_gltf_files(input_dir, output_dir)
    print("Parallel compression of all glb files complete!")