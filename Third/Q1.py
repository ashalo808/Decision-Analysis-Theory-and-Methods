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
print("原始数据:")
print(df)

# 1. 数据标准化
def normalize_matrix(matrix):
    # 对于效益型指标（产值、销售额、国家权益比重）和成本型指标（投资成本）分别处理
    # 效益型指标列索引: 0(产值), 2(销售额), 3(国家权益比重)
    # 成本型指标列索引: 1(投资成本)
    norm_matrix = np.zeros_like(matrix, dtype=float)
    
    # 处理效益型指标
    benefit_cols = [0, 2, 3]
    for col in benefit_cols:
        norm_matrix[:, col] = (matrix[:, col] - matrix[:, col].min()) / (matrix[:, col].max() - matrix[:, col].min())
    
    # 处理成本型指标
    cost_cols = [1]
    for col in cost_cols:
        norm_matrix[:, col] = (matrix[:, col].max() - matrix[:, col]) / (matrix[:, col].max() - matrix[:, col].min())
    
    return norm_matrix

matrix = df.values
norm_matrix = normalize_matrix(matrix)
print("\n标准化矩阵:")
print(pd.DataFrame(norm_matrix, index=df.index, columns=df.columns))

# 2. 离差最大化法确定权重
def deviation_maximization_weight(matrix):
    n, m = matrix.shape
    weights = np.zeros(m)
    
    for j in range(m):
        # 计算第j个属性的总离差
        total_deviation = 0
        for k in range(n):
            for l in range(n):
                total_deviation += abs(matrix[k, j] - matrix[l, j])
        
        weights[j] = total_deviation
    
    # 归一化权重
    weights = weights / weights.sum()
    return weights

weights = deviation_maximization_weight(norm_matrix)
print("\n属性权重:")
print(pd.Series(weights, index=df.columns))

# 3. TOPSIS方法
def topsis(matrix, weights, impact):
    # 加权标准化决策矩阵
    weighted_matrix = matrix * weights
    
    # 确定理想解和负理想解
    ideal_best = np.zeros(matrix.shape[1])
    ideal_worst = np.zeros(matrix.shape[1])
    
    for i in range(matrix.shape[1]):
        if impact[i] == 1:  # 效益型指标
            ideal_best[i] = weighted_matrix[:, i].max()
            ideal_worst[i] = weighted_matrix[:, i].min()
        else:  # 成本型指标
            ideal_best[i] = weighted_matrix[:, i].min()
            ideal_worst[i] = weighted_matrix[:, i].max()
    
    # 计算每个方案到理想解和负理想解的距离
    d_best = np.zeros(matrix.shape[0])
    d_worst = np.zeros(matrix.shape[0])
    
    for i in range(matrix.shape[0]):
        d_best[i] = np.sqrt(((weighted_matrix[i, :] - ideal_best) ** 2).sum())
        d_worst[i] = np.sqrt(((weighted_matrix[i, :] - ideal_worst) ** 2).sum())
    
    # 计算相对接近度
    closeness = d_worst / (d_best + d_worst)
    
    return closeness

# 定义指标类型 (1:效益型, 0:成本型)
impact = np.array([1, 0, 1, 1])  # 产值(效益), 投资成本(成本), 销售额(效益), 国家权益比重(效益)

closeness = topsis(norm_matrix, weights, impact)
ranking = pd.Series(closeness, index=df.index).sort_values(ascending=False)

print("\nTOPSIS结果:")
print("相对接近度:")
print(ranking)
print("\n最终排名:")
print(ranking.index.tolist())