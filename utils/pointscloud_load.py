import os
import h5py  # 使用 h5py 读取 MATLAB v7.3 文件
import numpy as np

# loader1: load from .mat file, and output format is (min_frames, points_num=128, input_size=6), this loader is for a simple test
def load_data(dataset_dir, num_points=128):
    data = []
    min_frames = float('inf')  # 初始为无穷大

    # 遍历文件夹以加载所有数据
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        if os.path.isdir(folder_path):
            mat_file = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
            if mat_file:
                with h5py.File(os.path.join(folder_path, mat_file[0]), 'r') as mat_data:
                    points = mat_data[next(iter(mat_data))][()]  # 访问第一个数据项
                    # 获取帧数
                    num_frames = points.shape[2]  # 174
                    if num_frames < min_frames:
                        min_frames = num_frames

    print(f"Minimum number of frames across all samples: {min_frames}")

    # 第二次遍历以加载数据并随机选择帧
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        if os.path.isdir(folder_path):
            mat_file = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
            if mat_file:
                with h5py.File(os.path.join(folder_path, mat_file[0]), 'r') as mat_data:
                    points = mat_data[next(iter(mat_data))][()]  # 访问第一个数据项

                # 随机选择 min_frames 帧
                num_frames = points.shape[2]
                selected_indices = np.random.choice(num_frames, min_frames, replace=False)
                points = points[:, :, selected_indices]  # 随机选择帧

                # 将数据展平为 (min_frames, 128, 6)
                points_flat = points.transpose(2, 0, 1)  # 变为 (min_frames, 128, 6)
                data.append(points_flat)

    # 在这里进行堆叠前检查数据的维度
    if data:
        print(f"Data shapes before stacking: {[d.shape for d in data]}")

    data_array = np.array(data)

    return data_array