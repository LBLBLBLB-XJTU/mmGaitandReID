import importlib  # 动态导入模块

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# 加载配置文件
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 加载配置
config = load_config('config.yaml')

# 获取配置
dataset_dir = config['dataset_dir']
loader_name = config['pointscloud_loader']
loader_file = config['pointscloud_loader_file']
input_size = config['input_size']  # 应该是6
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
loss_function = config['loss_function']
optimizer_type = config['optimizer']
model_name = config['model']
model_file = config['model_file']

# 使用对应的加载器加载数据
loader_module = importlib.import_module(loader_file)
loader = getattr(loader_module, loader_name)
X = loader(dataset_dir, num_points=128)

# 转换为 PyTorch 张量
X_tensor = torch.FloatTensor(X)

# 动态导入模型
model_module = importlib.import_module(model_file)  # 导入模型文件
model_class = getattr(model_module, model_name)  # 获取模型类
model = model_class(input_size)  # 实例化模型

# 根据配置选择损失函数
if loss_function == "BCEWithLogitsLoss":
    criterion = nn.BCEWithLogitsLoss()
else:
    raise ValueError(f"Unsupported loss function: {loss_function}")

# 根据配置选择优化器
if optimizer_type == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_type}")

# 划分训练集测试集：从 10 个样本中选择 7 个训练集样本和 3 个测试集样本
all_indices = np.arange(X_tensor.shape[0])  # 0 到 (num_samples - 1) 的索引
train_indices = np.random.choice(all_indices, 7, replace=False)
test_indices = np.setdiff1d(all_indices, train_indices)[:3]  # 剩下的 3 个作为测试集

# 训练模型
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 训练样本的标签均为 1
    train_samples = X_tensor[train_indices]  # 维度应为 (7, min_frames, 128, 6)
    labels = torch.ones(train_samples.size(0), 1)  # 所有标签为 1
    outputs = model(train_samples.view(-1, 128, 6))  # 调整形状为 (7 * min_frames, 128, 6)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_samples = X_tensor[test_indices]  # 测试样本
    test_outputs = model(test_samples.view(-1, 128, 6))  # 调整形状为 (3 * min_frames, 128, 6)
    predictions = torch.sigmoid(test_outputs)  # 将输出转换为概率

    # 判断是否为同一个人（阈值设置为 0.5）
    is_same_person = predictions > 0.5
    print(f'Test samples predictions: {predictions.squeeze().numpy()}')
    print(f'Is the selected samples the same person? {is_same_person.squeeze().numpy()}')