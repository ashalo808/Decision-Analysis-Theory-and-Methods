import numpy as np

# 转移矩阵
P_no_ad = np.array([[0.8, 0.2], [0.4, 0.6]])
P_ad = np.array([[0.9, 0.1], [0.7, 0.3]])

# 利润
profit = [100, 30]  # 畅销和滞销时的利润
ad_cost = 15        # 广告成本

# 初始状态：滞销
initial_state = 1  # 0=畅销, 1=滞销

# 模拟4年的决策
def simulate(P, years=4, initial=1, ad=False):
    current = initial
    total = 0
    for _ in range(years):
        total += profit[current] - (ad_cost if ad else 0)
        current = np.random.choice([0, 1], p=P[current])
    return total

# 蒙特卡洛模拟
n_simulations = 10000
ad_results = [simulate(P_ad, initial=initial_state, ad=True) for _ in range(n_simulations)]
no_ad_results = [simulate(P_no_ad, initial=initial_state, ad=False) for _ in range(n_simulations)]

# 计算期望利润
exp_ad = np.mean(ad_results)
exp_no_ad = np.mean(no_ad_results)

# 决策
should_ad = exp_ad > exp_no_ad

# 输出结果
print(f"不采用广告的4年期望利润: {exp_no_ad:.2f}万元")
print(f"采用广告的4年期望利润: {exp_ad:.2f}万元")
print(f"\n决策建议: {'应该' if should_ad else '不应该'}采用广告措施")
