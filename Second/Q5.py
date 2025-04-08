import numpy as np

def calculate_weights(matrix):
    """
    使用几何平均法计算权重向量
    参数:
        matrix: 判断矩阵
    返回:
        weights: 归一化的权重向量
    """
    n = matrix.shape[0]
    geometric_means = np.prod(matrix, axis=1) ** (1/n)
    return geometric_means / np.sum(geometric_means)

def check_consistency(matrix, weights):
    """
    检验判断矩阵的一致性
    参数:
        matrix: 判断矩阵
        weights: 权重向量
    返回:
        CR: 一致性比率
        is_consistent: 布尔值，表示矩阵是否一致
    """
    n = matrix.shape[0]
    # 计算最大特征值
    lambda_max = np.sum(np.dot(matrix, weights) / weights) / n
    
    # 计算一致性指标CI
    CI = (lambda_max - n) / (n - 1)
    
    # 随机一致性指标RI
    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_table.get(n, 1.5)  # 默认值为1.5
    
    # 一致性比率CR
    CR = CI / RI if RI != 0 else 0
    
    # 判断是否满足一致性要求(CR < 0.1)
    is_consistent = CR < 0.1
    
    return CR, is_consistent

def ahp_analysis():
    """实现层次分析法(AHP)来评估三个候选人"""
    # 定义属性名称
    attributes = ["健康状况", "业务知识", "书面表达能力", "口才", "道德水平", "工作作风", "人际关系"]
    
    # 属性重要性矩阵A
    A = np.array([
        [1, 1, 1, 2, 4, 1, 1/2],
        [1, 1, 1/2, 1, 5, 3, 1/2],
        [1, 2, 1, 1/4, 1/5, 1, 1/3],
        [1/2, 1, 4, 1, 1/3, 3, 1],
        [1/4, 1/5, 5, 3, 1, 1, 1],
        [1, 2, 3, 1/3, 1, 1, 1],
        [2, 2, 3, 1, 1, 1, 1]
    ])
    
    # 计算属性权重
    weights = calculate_weights(A)
    
    # 检验一致性
    CR, is_consistent = check_consistency(A, weights)
    if not is_consistent:
        print(f"警告：属性重要性矩阵一致性比率CR={CR:.4f} > 0.1，判断矩阵不一致，可能需要重新评估!")
    else:
        print(f"属性重要性矩阵一致性比率CR={CR:.4f} < 0.1，判断一致。")
    
    # 候选人在各属性下的判断矩阵
    candidate_matrices = [
        # 健康状况
        np.array([
            [1, 1/4, 1/3],
            [4, 1, 3],
            [3, 1/3, 1]
        ]),
        # 业务知识
        np.array([
            [1, 1/4, 1/2],
            [4, 1, 3],
            [2, 1/3, 1]
        ]),
        # 书面表达能力
        np.array([
            [1, 4, 2],
            [1/4, 1, 1/3],
            [1/2, 3, 1]
        ]),
        # 口才
        np.array([
            [1, 1/3, 1/5],
            [3, 1, 1/2],
            [5, 2, 1]
        ]),
        # 道德水平
        np.array([
            [1, 2, 1/3],
            [1/2, 1, 1/4],
            [3, 4, 1]
        ]),
        # 工作作风
        np.array([
            [1, 1/2, 1/4],
            [2, 1, 1/3],
            [4, 3, 1]
        ]),
        # 人际关系
        np.array([
            [1, 1/3, 1/2],
            [3, 1, 2],
            [2, 1/2, 1]
        ])
    ]
    
    # 计算每个属性下候选人的权重并检验一致性
    candidate_weights = np.zeros((len(attributes), 3))
    for i in range(len(attributes)):
        mat = candidate_matrices[i]
        candidate_weights[i] = calculate_weights(mat)
        
        CR, is_consistent = check_consistency(mat, candidate_weights[i])
        if not is_consistent:
            print(f"警告：{attributes[i]}判断矩阵一致性比率CR={CR:.4f} > 0.1，判断不一致!")
        else:
            print(f"{attributes[i]}判断矩阵一致性比率CR={CR:.4f} < 0.1，判断一致。")
    
    # 综合评分
    total_scores = np.dot(weights, candidate_weights)
    
    # 输出结果
    print("\n属性权重:")
    for i, attr in enumerate(attributes):
        print(f"{attr}: {weights[i]:.4f}")
    
    print("\n候选人在各属性下的权重:")
    for i, attr in enumerate(attributes):
        print(f"{attr}: X={candidate_weights[i][0]:.4f}, Y={candidate_weights[i][1]:.4f}, Z={candidate_weights[i][2]:.4f}")
    
    print("\n综合评分:")
    print(f"X: {total_scores[0]:.4f}")
    print(f"Y: {total_scores[1]:.4f}")
    print(f"Z: {total_scores[2]:.4f}")
    
    best_candidate = ['X', 'Y', 'Z'][np.argmax(total_scores)]
    print(f"\n最佳候选人是: {best_candidate}")

if __name__ == "__main__":
    ahp_analysis()
