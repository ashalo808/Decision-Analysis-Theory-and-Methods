import numpy as np

# 先验概率
P_theta = np.array([0.3, 0.4, 0.3])

# 条件概率矩阵 P(H|theta)
P_H_theta = np.array([
    [0.6, 0.2, 0.2],  # H1
    [0.3, 0.5, 0.2],  # H2
    [0.1, 0.3, 0.6]   # H3
])

# 联合概率 P(H, theta) = P(H|theta)*P(theta)
P_joint = P_H_theta * P_theta

# 边际概率 P(H)
P_H = P_joint.sum(axis=1)

# 后验概率 P(theta|H) = P(H,theta)/P(H)
P_theta_H = (P_joint.T / P_H).T

# 收益矩阵
profit = np.array([
    [50, 20, -20],
    [30, 25, -10],
    [10, 10, 10]
])

# 计算每种调查结果下的最优决策
results = {}
for i, H in enumerate(['H1', 'H2', 'H3']):
    # 计算期望收益
    exp_profit = profit @ P_theta_H[i]
    best_choice = np.argmax(exp_profit)
    results[H] = {
        '后验概率': P_theta_H[i],
        '期望收益': exp_profit,
        '最优决策': ['大型设备', '中型设备', '小型设备'][best_choice],
        '最大期望收益': exp_profit[best_choice]
    }

# 输出结果
print("后验概率分布:")
for H in results:
    print(f"{H}: θ1={results[H]['后验概率'][0]:.3f}, θ2={results[H]['后验概率'][1]:.3f}, θ3={results[H]['后验概率'][2]:.3f}")

print("\n各调查结果下的最优决策:")
for H in results:
    print(f"当调查结果为{H}时:")
    print(f"  期望收益: 大型={results[H]['期望收益'][0]:.2f}, 中型={results[H]['期望收益'][1]:.2f}, 小型={results[H]['期望收益'][2]:.2f}")
    print(f"  最优决策: {results[H]['最优决策']}, 期望收益={results[H]['最大期望收益']:.2f}万元")

# 计算整体期望收益
total_exp_profit = sum(results[H]['最大期望收益'] * P_H[i] for i, H in enumerate(results))
print(f"\n考虑调查后的总体期望收益: {total_exp_profit:.2f}万元")
