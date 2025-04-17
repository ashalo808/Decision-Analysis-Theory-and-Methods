import numpy as np
import pandas as pd

# 1. 准备数据
# 决策矩阵
decision_matrix = np.array([
    [0.800, 0.752, 0.765, 0.758],  # C1
    [0.800, 0.644, 0.634, 0.675],  # C2
    [0.646, 0.589, 0.244, 0.545],  # C3
    [0.707, 0.007, 0.105, -0.197], # C4
    [0.220, -0.337, 0.086, -0.016],# C5
    [0.686, 0.689, 0.609, 0.605],  # C6
    [0.707, 0.580, 0.436, 0.405],  # C7
    [0.232, -0.009, -0.277, -0.034],# C8
    [0.589, 0.465, 0.339, -0.155], # C9
    [0.734, 0.547, 0.534, 0.682],  # C10
    [0.619, 0.734, -0.036, -0.129],# C11
    [-0.028, 0.439, 0.246, 0.642], # C12
    [0.639, -0.129, 0.240, 0.666], # C13
    [0.621, 0.466, 0.389, 0.707],  # C14
    [0.007, 0.372, -0.056, -0.057],# C15
    [-0.033, -0.016, 0.056, 0.004] # C16
])

# 阈值表
q = np.array([0.11, 0.11, -0.33, 0.11, 0.11, -0.33, -0.33, 0.11, -0.33, -0.33, 0.11, -0.33, -0.33, 0.11, 0.11, 0.11])  # 无差异阈值
p = np.array([0.47, 0.47, 0.11, 0.47, 0.47, 0.11, 0.11, 0.47, 0.11, -0.33, 0.47, 0.11, 0.11, 0.47, 0.47, 0.47])  # 偏好阈值
v = np.array([0.81, 0.81, 0.70, 0.81, 0.81, 0.70, 0.70, 0.81, 0.70, 0.80, 0.70, 0.70, 0.81, 0.81, 0.81, 0.81])  # 否决阈值

# 属性权重
weights = np.array([0.080, 0.078, 0.057, 0.050, 0.138, 0.068, 0.054, 0.054, 0.048, 0.051, 0.075, 0.048, 0.048, 0.054, 0.052, 0.045])

alternatives = ['A1', 'A2', 'A3', 'A4']
criteria = [f'C{i+1}' for i in range(16)]

# 2. ELECTRE III 实现
def electre_iii(decision_matrix, weights, q, p, v):
    n_alternatives = decision_matrix.shape[1]
    n_criteria = decision_matrix.shape[0]
    
    # 计算和谐指数
    concordance = np.zeros((n_alternatives, n_alternatives))
    
    for i in range(n_alternatives):
        for j in range(n_alternatives):
            if i == j:
                continue
            sum_weights = 0
            for k in range(n_criteria):
                # 计算每个准则下的和谐程度
                diff = decision_matrix[k, i] - decision_matrix[k, j]
                if diff >= -q[k]:
                    c_k = 1
                elif diff <= -p[k]:
                    c_k = 0
                else:
                    c_k = (p[k] + diff) / (p[k] - q[k])
                sum_weights += weights[k] * c_k
            concordance[i, j] = sum_weights / weights.sum()
    
    # 计算不和谐指数
    discordance = np.zeros((n_criteria, n_alternatives, n_alternatives))
    
    for k in range(n_criteria):
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i == j:
                    continue
                diff = decision_matrix[k, j] - decision_matrix[k, i]
                if diff <= p[k]:
                    d_k = 0
                elif diff >= v[k]:
                    d_k = 1
                else:
                    d_k = (diff - p[k]) / (v[k] - p[k])
                discordance[k, i, j] = d_k
    
    # 计算可信度
    credibility = np.zeros((n_alternatives, n_alternatives))
    
    for i in range(n_alternatives):
        for j in range(n_alternatives):
            if i == j:
                continue
            credibility[i, j] = concordance[i, j]
            for k in range(n_criteria):
                if discordance[k, i, j] > concordance[i, j]:
                    credibility[i, j] *= (1 - discordance[k, i, j]) / (1 - concordance[i, j])
    
    # 确定级别优先关系
    lambda_cut = 0.6  # 通常设为0.6-0.8
    outranking = credibility >= lambda_cut
    
    # 蒸馏和反蒸馏过程（简化版）
    # 这里简化为计算净流量
    positive_flow = outranking.sum(axis=1)
    negative_flow = outranking.sum(axis=0)
    net_flow = positive_flow - negative_flow
    
    return net_flow

# 3. 执行ELECTRE III并获取排序
net_flows = electre_iii(decision_matrix, weights, q, p, v)
ranking = pd.Series(net_flows, index=alternatives).sort_values(ascending=False)

print("ELECTRE III 结果:")
print("净流量值:")
print(ranking)
print("\n最终排名:")
print(ranking.index.tolist())