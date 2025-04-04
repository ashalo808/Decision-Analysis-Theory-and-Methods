import pandas as pd

# 定义参数
unit_profit = 24  # 售出时的单位利润
unit_loss = 6     # 未售出时的单位损失
sales_distribution = {
    7000: 0.2,
    8000: 0.4,
    9000: 0.3,
    10000: 0.1
}

# 计算各生产量下的期望利润
def calculate_expected_profit(production):
    expected_profit = 0
    for sales, prob in sales_distribution.items():
        if sales >= production:
            # 全部能卖出
            profit = production * unit_profit
        else:
            # 只能卖出sales个，剩余(production - sales)个会有损失
            profit = sales * unit_profit - (production - sales) * unit_loss
        expected_profit += profit * prob
    return expected_profit

# 评估所有可能的生产量
production_levels = [7000, 8000, 9000, 10000]
results = []

for prod in production_levels:
    exp_profit = calculate_expected_profit(prod)
    results.append({
        '生产量(个)': prod,
        '期望利润(元)': round(exp_profit, 2)
    })

# 创建DataFrame显示结果
df = pd.DataFrame(results)
print(df)

# 找出最优生产量
optimal = df.loc[df['期望利润(元)'].idxmax()]
print(f"\n最优生产量为{optimal['生产量(个)']}个，期望利润为{optimal['期望利润(元)']}元")
