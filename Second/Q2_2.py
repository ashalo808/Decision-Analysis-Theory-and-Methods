import numpy as np
from numpy.linalg import eig
from tqdm import tqdm

def generate_random_matrix(n):
    """生成随机判断矩阵"""
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # 随机生成1-9或其倒数
            value = np.random.choice([1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 
                                     1, 2, 3, 4, 5, 6, 7, 8, 9])
            matrix[i, j] = value
            matrix[j, i] = 1 / value
    return matrix

def calculate_ci(matrix):
    """计算一致性指标CI"""
    n = matrix.shape[0]
    eigenvalues = eig(matrix)[0]
    lambda_max = max(eigenvalues.real)
    CI = (lambda_max - n) / (n - 1)
    return CI

def simulate_ri(n, num_samples=1000, method='eigen'):
    """模拟计算随机一致性指标RI"""
    CI_values = []
    for _ in tqdm(range(num_samples), desc=f"n={n}"):
        matrix = generate_random_matrix(n)
        if method == 'eigen':
            CI = calculate_ci(matrix)
        elif method == 'root':
            # 方根法计算CI
            roots = np.power(np.prod(matrix, axis=1), 1/n)
            weights = roots / roots.sum()
            weighted_sum = matrix @ weights
            lambda_max = (weighted_sum / weights).mean()
            CI = (lambda_max - n) / (n - 1)
        CI_values.append(CI)
    return np.mean(CI_values)

# 验证不同阶数的RI值
max_n = 9  # 最大矩阵阶数
num_samples = 1000  # 每个阶数的样本数

print("特征值法计算的RI值:")
ri_eigen = [0]  # n=1时RI=0
for n in range(2, max_n+1):
    ri = simulate_ri(n, num_samples, method='eigen')
    ri_eigen.append(ri)
print("n=1-9的RI值(特征值法):", [f"{x:.4f}" for x in ri_eigen])

print("\n方根法计算的RI值:")
ri_root = [0]  # n=1时RI=0
for n in range(2, max_n+1):
    ri = simulate_ri(n, num_samples, method='root')
    ri_root.append(ri)
print("n=1-9的RI值(方根法):", [f"{x:.4f}" for x in ri_root])

# 与参考值对比
ref_eigen = [0, 0.5182, 0.8942, 1.1104, 1.2480, 1.3324, 1.4028, 1.4520, 1.4844]
ref_root = [0, 0.5230, 0.8601, 1.0835, 1.2228, 1.3189, 1.3929, 1.4381, 1.4717]

print("\n与参考值对比(特征值法):")
for n in range(1, max_n+1):
    print(f"n={n}: 模拟值={ri_eigen[n-1]:.4f}, 参考值={ref_eigen[n-1]:.4f}")

print("\n与参考值对比(方根法):")
for n in range(1, max_n+1):
    print(f"n={n}: 模拟值={ri_root[n-1]:.4f}, 参考值={ref_root[n-1]:.4f}")
