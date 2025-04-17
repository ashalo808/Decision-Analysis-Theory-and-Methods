import numpy as np
from numpy.linalg import eig

# 给定的判断矩阵
A = np.array([
    [1, 1, 1, 4, 1, 1/2],
    [1, 1, 2, 4, 1, 1/2],
    [1, 1/2, 1, 5, 3, 1/2],
    [1/4, 1/4, 1/5, 1, 1/3, 1/3],
    [1, 1, 1/3, 3, 1, 1],
    [2, 2, 2, 3, 1, 1]
])

# 1. 求和法
def sum_method(matrix):
    # 列归一化
    norm_matrix = matrix / matrix.sum(axis=0)
    # 行求和平均
    weights = norm_matrix.mean(axis=1)
    return weights

# 2. 方根法
def root_method(matrix):
    # 计算几何平均
    row_products = np.prod(matrix, axis=1)
    roots = np.power(row_products, 1/matrix.shape[0])
    # 归一化
    weights = roots / roots.sum()
    return weights

# 3. 特征值法
def eigenvalue_method(matrix):
    # 求特征值与特征向量
    eigenvalues, eigenvectors = eig(matrix)
    
    max_index = np.argmax(eigenvalues)
    max_eigenvector = eigenvectors[:, max_index].real
    
    # 计算权重
    weights = max_eigenvector / max_eigenvector.sum()
    return weights

# 一致性检验
def consistency_check(matrix, weights):
    n = matrix.shape[0]
    # 计算最大特征值
    weighted_sum = matrix @ weights
    lambda_max = (weighted_sum / weights).mean()
    # 计算CI
    CI = (lambda_max - n) / (n - 1)
    # RI值(对于n=6)
    RI = 1.24  
    # 计算CR
    CR = CI / RI
    return lambda_max, CI, CR

# 计算权重
weights_sum = sum_method(A)
weights_root = root_method(A)
weights_eigen = eigenvalue_method(A)

# 一致性检验
lambda_max_sum, CI_sum, CR_sum = consistency_check(A, weights_sum)
lambda_max_root, CI_root, CR_root = consistency_check(A, weights_root)
lambda_max_eigen, CI_eigen, CR_eigen = consistency_check(A, weights_eigen)

# 打印结果
print("求和法权重:", weights_sum)
print("方根法权重:", weights_root)
print("特征值法权重:", weights_eigen)
print("\n一致性检验结果:")
print(f"求和法: λ_max={lambda_max_sum:.4f}, CI={CI_sum:.4f}, CR={CR_sum:.4f}")
print(f"方根法: λ_max={lambda_max_root:.4f}, CI={CI_root:.4f}, CR={CR_root:.4f}")
print(f"特征值法: λ_max={lambda_max_eigen:.4f}, CI={CI_eigen:.4f}, CR={CR_eigen:.4f}")

if CR_sum < 0.1 and CR_root < 0.1 and CR_eigen < 0.1:
    print("\n所有方法的CR值均小于0.1，判断矩阵具有满意的一致性")
else:
    print("\n警告: CR值大于0.1，判断矩阵一致性不满足要求")