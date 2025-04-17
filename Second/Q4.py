import numpy as np
import pandas as pd

# 原始数据
data = {
    '企业': ['a1', 'a2', 'a3', 'a4'],
    '产值（万元）': [8350, 7455, 11000, 9624],
    '投资成本（万元）': [5300, 4952, 8001, 5000],
    '销售额（万元）': [6135, 6527, 9008, 8892],
    '国家权益比重': [0.82, 0.65, 0.59, 0.74]
}

df = pd.DataFrame(data).set_index('企业')

# 数据标准化
def normalize_matrix(matrix):
    # 对于效益型指标（越大越好）和成本型指标（越小越好）分别处理
    # 这里假设产值、销售额、国家权益比重是效益型指标，投资成本是成本型指标
    norm_matrix = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        if j == 1:  # 投资成本是成本型指标
            min_val = np.min(matrix[:, j])
            norm_matrix[:, j] = min_val / matrix[:, j]
        else:  # 其他是效益型指标
            max_val = np.max(matrix[:, j])
            norm_matrix[:, j] = matrix[:, j] / max_val
    return norm_matrix

matrix = df.values
norm_matrix = normalize_matrix(matrix)

# 熵权法计算权重
def entropy_weight(matrix):
    # 计算每个指标的比重
    p = matrix / np.sum(matrix, axis=0)
    
    # 计算每个指标的熵值
    epsilon = 1e-12  # 避免log(0)
    e = -np.sum(p * np.log(p + epsilon), axis=0) / np.log(len(matrix))
    
    # 计算差异系数
    g = 1 - e
    
    # 计算权重
    weights = g / np.sum(g)
    return weights

entropy_weights = entropy_weight(norm_matrix)
print("熵权法计算的权重:")
for col, weight in zip(df.columns, entropy_weights):
    print(f"{col}: {weight:.4f}")
    
def deviation_maximization(matrix):
    # 计算每个指标下各方案与其他方案的距离
    n, m = matrix.shape
    total_deviation = np.zeros(m)
    
    for j in range(m):
        for i in range(n):
            for k in range(n):
                total_deviation[j] += abs(matrix[i, j] - matrix[k, j])
    
    # 计算权重
    weights = total_deviation / np.sum(total_deviation)
    return weights

deviation_weights = deviation_maximization(norm_matrix)
print("\n离差最大化方法计算的权重:")
for col, weight in zip(df.columns, deviation_weights):
    print(f"{col}: {weight:.4f}")
