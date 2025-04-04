"""
使用后悔值准则，进行决策
"""
import numpy as np

# 计算后悔值
def calculate_regrets(matrix):
    regrets = np.zeros_like(matrix)
    for j in range(len(matrix[0])):
        # 找到每一列的最大值
        max_val = np.max(matrix[:, j])
        for i in range(len(matrix)):
            # 计算后悔值
            regrets[i][j] = max_val - matrix[i][j]
    return regrets

# 使用后悔值准则决策
def regret_criterion(regrets):
    # 计算每个方案的最大后悔值
    max_regrets = np.max(regrets, axis=1)
    # 选择最大后悔值最小的方案
    best_choice = np.argmin(max_regrets)
    return best_choice, max_regrets

# 根据题目,直接设置收益矩阵
def example():
    # 三种方案，四种市场需求状态
    payoffs_matrix = np.array([
        [152, 70, -60, -150],  # 方案A1 (建设新厂)
        [100, 45, -13, -53],  # 方案A2 (技术改造)
        [118, 63, -45, -98]   # 方案A3 (扩建部分工厂)
    ])
    
    print("收益矩阵:")
    print(payoffs_matrix)
    
    # 计算后悔值
    regrets = calculate_regrets(payoffs_matrix)
    
    # 使用后悔值准则做决策
    best_choice, max_regrets = regret_criterion(regrets)
    
    # 输出结果
    print("\n后悔值矩阵:")
    print(regrets)
    print("\n各方案的最大后悔值:")
    for i in range(len(max_regrets)):
        print(f"方案 {i+1}: {max_regrets[i]}")
    print(f"\n最优决策: 方案 {best_choice+1}")

# 调用函数
def main():
    # 手动输入收益矩阵
    print("请输入方案数量 m:")
    m = int(input())
    print("请输入状态数量 n:")
    n = int(input())
    
    # 初始化收益矩阵
    payoffs_matrix = np.zeros((m, n))
    
    for i in range(0, m):
        for j in range(0, n):
            payoffs_matrix[i][j] = int(input())
            
    # 计算后悔值
    regrets = calculate_regrets(payoffs_matrix)
    
    # 使用后悔值准则做决策
    best_choice, max_regrets = regret_criterion(regrets)
    
    # 输出结果
    print("\n原始收益矩阵:")
    print(payoffs_matrix)
    print("\n后悔值矩阵:")
    print(regrets)
    print("\n各方案的最大后悔值:")
    for i in range(len(max_regrets)):
        print(f"方案 {i+1}: {max_regrets[i]}")
    print(f"\n最优决策: 方案 {best_choice+1}")

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 手动输入数据")
    print("2. 使用题目给定的例子")
    choice = int(input())
    
    if choice == 1:
        main()
    else:
        example()
    