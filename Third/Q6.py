import numpy as np

# 定义属性权重和专家权重
w = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.2, 0.15])
a = np.array([0.27, 0.23, 0.24, 0.26])

# 决策矩阵（表1.4和表1.5）
R1 = np.array([
    [85, 90, 95, 60, 70, 80, 90, 85],
    [95, 80, 60, 70, 90, 85, 80, 70],
    [65, 75, 95, 65, 90, 95, 70, 85],
    [75, 75, 50, 65, 95, 75, 85, 80]
])

R2 = np.array([
    [60, 75, 90, 65, 70, 95, 70, 75],
    [85, 60, 60, 65, 90, 75, 95, 70],
    [60, 65, 75, 80, 90, 95, 90, 80],
    [65, 60, 60, 70, 90, 85, 70, 65]
])

# 假设表1.6和表1.7的数据（题目未提供，这里随机生成）
R3 = np.random.randint(50, 100, size=(4, 8))
R4 = np.random.randint(50, 100, size=(4, 8))

# 方法一：加权平均算子
def weighted_average_method(R_list, w, a):
    # 规范化
    normalized_R = []
    for R in R_list:
        max_values = np.max(R, axis=0)
        normalized_R.append(R / max_values)
    
    # 加权平均
    weighted_scores = np.zeros(4)
    for i in range(4):
        for j in range(8):
            weighted_scores[i] += a[i] * np.sum(w[j] * normalized_R[i][:, j])
    
    # 排序
    ranking = np.argsort(-weighted_scores)
    return weighted_scores, ranking

# 方法二：Borda函数
def borda_method(R_list):
    borda_scores = np.zeros(4)
    for R in R_list:
        for j in range(8):
            ranks = np.argsort(-R[:, j])  # 从高到低排序
            for rank, idx in enumerate(ranks):
                borda_scores[idx] += (3 - rank)  # 第一名3分，第二名2分，第三名1分，第四名0分
    # 排序
    ranking = np.argsort(-borda_scores)
    return borda_scores, ranking

# 执行方法一
weighted_scores, weighted_ranking = weighted_average_method([R1, R2, R3, R4], w, a)
print("加权平均方法：")
print("综合评分：", weighted_scores)
print("排序（从高到低）：", weighted_ranking + 1)  # 方案编号从1开始

# 执行方法二
borda_scores, borda_ranking = borda_method([R1, R2, R3, R4])
print("\nBorda方法：")
print("Borda总分：", borda_scores)
print("排序（从高到低）：", borda_ranking + 1)