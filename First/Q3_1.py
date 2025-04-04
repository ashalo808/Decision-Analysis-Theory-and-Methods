import numpy as np

# 收益矩阵
profit_matrix = np.array([
    [50, 20, -20],
    [30, 25, -10],
    [10, 10, 10]
])

# 各状态的概率
prob = np.array([0.3, 0.4, 0.3])

# 计算每个方案的期望收益
expected_profits = np.dot(profit_matrix, prob)

# 找出最优方案
optimal_index = np.argmax(expected_profits)
optimal_profit = expected_profits[optimal_index]
options = ['引进大型设备', '引进中型设备', '引进小型设备']
optimal_option = options[optimal_index]

# 输出结果
print("各方案的期望收益：")
for i in range(len(options)):
    print(f"{options[i]}: {expected_profits[i]:.1f}万元")

print(f"\n最优方案是: {optimal_option}, 期望收益为: {optimal_profit:.1f}万元")
