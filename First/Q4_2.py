import numpy as np

# 转移矩阵
P_no_ad = np.array([[0.8, 0.2], [0.4, 0.6]])
P_ad = np.array([[0.9, 0.1], [0.7, 0.3]])

# 利润
profit = np.array([100, 30])  # 畅销和滞销时的利润
ad_cost = 15        # 广告成本

# 初始状态：滞销
# 正确表示为概率分布 [0, 1]，表示有100%概率处于滞销状态
initial_state = np.array([0, 1])  # 0=畅销, 1=滞销

# 使用马尔可夫链计算期望利润，并返回每年的状态概率
def calculate_expected_profit(P, years=4, initial_state=np.array([0, 1]), ad=False):
    state = initial_state.copy()  # 初始状态向量
    total_profit = 0
    
    yearly_states = [initial_state.copy()]  # 记录每年的状态概率
    yearly_profits = []  # 记录每年的期望利润
    
    for year in range(years):
        # 状态转移 (先转移状态，再计算当年利润)
        state = state @ P
        yearly_states.append(state.copy())
        
        # 计算当前状态的期望利润: 状态概率 * 对应利润
        current_profit = state[0] * profit[0] + state[1] * profit[1]
        if ad:
            current_profit -= ad_cost
            
        total_profit += current_profit
        yearly_profits.append(current_profit)
    
    return total_profit, yearly_states, yearly_profits

# 计算期望利润和每年状态概率
exp_no_ad, states_no_ad, profits_no_ad = calculate_expected_profit(P_no_ad, initial_state=initial_state, ad=False)
exp_ad, states_ad, profits_ad = calculate_expected_profit(P_ad, initial_state=initial_state, ad=True)

# 决策
should_ad = exp_ad > exp_no_ad

# 输出结果
print(f"不采用广告的4年期望利润: {exp_no_ad:.2f}万元")
print(f"采用广告的4年期望利润: {exp_ad:.2f}万元")
print(f"\n决策建议: {'应该' if should_ad else '不应该'}采用广告措施")

# 输出每年的状态概率分布和利润
print("\n每年状态概率分布和利润详情:")
print("=" * 75)
print("年份   |      不采用广告        |           采用广告           |")
print("      | 畅销概率 滞销概率 期望利润 | 畅销概率 滞销概率 期望利润    |")
print("-" * 75)

# 初始年份没有利润，只有状态
initial_no_ad = states_no_ad[0]
initial_ad = states_ad[0]
print(f"初始   | {initial_no_ad[0]:.4f}  {initial_no_ad[1]:.4f}     -   | {initial_ad[0]:.4f}  {initial_ad[1]:.4f}     -   |")

# 第1-4年有状态和利润
for year in range(1, 5):  # 第1年到第4年
    # 不采用广告的数据 (注意：利润索引是year-1，因为profits数组从0开始)
    no_ad_state = states_no_ad[year]
    no_ad_profit = profits_no_ad[year-1]
    
    # 采用广告的数据
    ad_state = states_ad[year]
    ad_profit = profits_ad[year-1]
    
    print(f"第{year}年  | {no_ad_state[0]:.4f}  {no_ad_state[1]:.4f}  {no_ad_profit:6.2f} | {ad_state[0]:.4f}  {ad_state[1]:.4f}  {ad_profit:6.2f} |")

print("=" * 75)
