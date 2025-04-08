import numpy as np
from scipy.optimize import minimize

def calculate_bwm_weights(BO, OW, criteria_names):
    """
    计算BWM权重
    :param BO: Best-to-Others向量
    :param OW: Others-to-Worst向量
    :param criteria_names: 准则名称列表
    :return: 权重向量和一致性比率
    """
    n = len(criteria_names)
    
    # 构建优化问题
    def objective(x):
        weights = x[:-1]    # 前 n 个元素为权重
        xi = x[-1]          # 最有一个元素为 ξ (最大偏差)
        
        # 计算约束违反程度
        max_violation = 0
        
        # 最佳准则对其他准则的约束
        best_idx = BO.index(1)  # 最佳准则的索引(BO中值为1的位置)
        for j in range(n):
            if j != best_idx and BO[j] != 0:  # 跳过最佳准则自身
                violation = abs(weights[best_idx]/weights[j] - BO[j])
                max_violation = max(max_violation, violation)
        
        # 其他准则对最差准则的约束
        worst_idx = OW.index(1)  # 最差准则的索引(OW中值为1的位置)
        for j in range(n):
            if j != worst_idx and OW[j] != 0:  # 跳过最差准则自身
                violation = abs(weights[j]/weights[worst_idx] - OW[j])
                max_violation = max(max_violation, violation)
        
        # 目标是使ξ等于最大违反程度，但不小于它
        return max(xi, max_violation)
    
    # 初始猜测(均匀权重 + 小的ξ值)
    x0 = np.ones(n+1) / (n+1)
    
    # 约束条件: 权重和为1，所有权重大于0，ξ大于0
    def weights_sum_to_one(x):
        return sum(x[:-1]) - 1
    
    constraints = ({
        'type': 'eq', 
        'fun': weights_sum_to_one
    })
    
    # 边界条件: 权重和ξ都非负
    bounds = [(0.001, 1) for _ in range(n)] + [(0, None)]
    
    # 优化求解
    result = minimize(
        objective, 
        x0, 
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'disp': False}
    )
    
    # 提取结果
    weights = result.x[:-1]
    xi_star = result.x[-1]
    
    # 计算一致性指标CI (根据准则数量)
    consistency_index = {
        2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11,
        6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45, 10: 1.49
    }
    CI = consistency_index.get(n, 1.5)  # 默认为1.5如果超出范围
    
    # 计算一致性比率
    CR = xi_star / CI if CI > 0 else 0
    
    # 输出结果
    print("\n最佳最差法(BWM)求解结果:")
    print("=" * 50)
    for i, name in enumerate(criteria_names):
        print(f"{name}: {weights[i]:.4f}")
    print("-" * 50)
    print(f"一致性比率(CR): {CR:.4f}")
    if CR < 0.1:
        print("一致性良好 (CR < 0.1)")
    else:
        print("警告: 一致性不佳 (CR >= 0.1)，建议重新评估比较数据")
    print("=" * 50)
    
    return weights, CR

# 示例使用
if __name__ == "__main__":
    # 定义准则名称
    criteria = ["Quality", "Price", "Comfort", "Safety", "Style"]
    
    # 示例1数据
    print("\n计算Example 1:")
    BO1 = [2, 1, 4, 2, 8]  # Best-to-Others (Best criterion: Price)
    OW1 = [4, 8, 2, 4, 1]  # Others-to-Worst (Worst criterion: Style)
    weights1, cr1 = calculate_bwm_weights(BO1, OW1, criteria)
    
    # 示例2数据
    print("\n计算Example 2:")
    BO2 = [2, 1, 4, 3, 8]  # Best-to-Others (Best criterion: Price)
    OW2 = [4, 8, 2, 3, 1]  # Others-to-Worst (Worst criterion: Style)
    weights2, cr2 = calculate_bwm_weights(BO2, OW2, criteria)
