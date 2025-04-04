import numpy as np

# 转移矩阵
P1 = np.array([[0.80, 0.15, 0.05], [0.20, 0.45, 0.35], [0.30, 0.40, 0.30]])
P2 = np.array([[0.90, 0.05, 0.05], [0.15, 0.75, 0.10], [0.10, 0.15, 0.75]])
P3 = np.array([[0.90, 0.05, 0.05], [0.10, 0.80, 0.10], [0.10, 0.15, 0.75]])

# 初始市场份额（假设均等）
initial = np.array([1/3, 1/3, 1/3])

# 计算稳态分布
def steady_state(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    steady = eigvecs[:, np.isclose(eigvals, 1)][:, 0]
    steady = steady / steady.sum()
    return steady.real

steady1 = steady_state(P1)
steady2 = steady_state(P2)
steady3 = steady_state(P3)

# 总销量和利润
total_sales = 1000  # 万件
profit_per = 1      # 元/件
costs = [150, 40, 30]  # 万

# 计算长期利润
long_term_profit = [
    steady1[0] * total_sales * profit_per - costs[0],
    steady2[0] * total_sales * profit_per - costs[1],
    steady3[0] * total_sales * profit_per - costs[2]
]

# 最优选择
best = np.argmax(long_term_profit)
options = ["发放债券", "广告宣传", "优质服务"]

# 输出结果
print("稳态市场份额分布:")
print(f"方案1: A={steady1[0]:.3f}, B={steady1[1]:.3f}, C={steady1[2]:.3f}")
print(f"方案2: A={steady2[0]:.3f}, B={steady2[1]:.3f}, C={steady2[2]:.3f}")
print(f"方案3: A={steady3[0]:.3f}, B={steady3[1]:.3f}, C={steady3[2]:.3f}")

print("\n长期利润(万元):")
for i in range(3):
    print(f"{options[i]}: {long_term_profit[i]:.2f}")

print(f"\n最优方案是: {options[best]}, 长期利润为: {long_term_profit[best]:.2f}万元")
