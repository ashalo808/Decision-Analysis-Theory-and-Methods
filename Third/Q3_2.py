# 3-2 α法求解多目标优化问题
from scipy.optimize import linprog

def alpha_method():
    # 首先需要确定每个目标的理想值
    
    # 求解f1的最小值
    c_f1 = [4, -6]  # min 4x1 - 6x2
    A = [[2, 4], [6, 3]]
    b = [14, 24]
    bounds = [(0, None), (0, None)]
    res_f1 = linprog(c_f1, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    f1_min = res_f1.fun
    
    # 求解f2的最大值
    c_f2 = [-3, -3]  # max 3x1 + 3x2 → min -3x1 - 3x2
    res_f2 = linprog(c_f2, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    f2_max = -res_f2.fun
    
    print(f"\n各目标理想值:")
    print(f"f1最小值 = {f1_min:.2f}")
    print(f"f2最大值 = {f2_max:.2f}")
    
    # 使用α法求解
    alpha = 0.5  # 可以调整α值来探索不同解
    
    # 构建新的目标函数
    # 归一化目标并加权求和
    # 我们需要估计f1的最大值和f2的最小值来归一化
    # 估计f1的最大值
    c_f1_max = [-4, 6]  # max 4x1 - 6x2 → min -4x1 + 6x2
    res_f1_max = linprog(c_f1_max, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    f1_max = -res_f1_max.fun
    
    # 估计f2的最小值
    c_f2_min = [3, 3]  # min 3x1 + 3x2
    res_f2_min = linprog(c_f2_min, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    f2_min = res_f2_min.fun
    
    print(f"f1最大值估计 = {f1_max:.2f}")
    print(f"f2最小值估计 = {f2_min:.2f}")
    
    # 归一化后的目标函数
    # 目标: min α*(f1-f1_min)/(f1_max-f1_min) + (1-α)*(f2_max-f2)/(f2_max-f2_min)
    # 转换为线性规划可解的形式
    
    # 由于目标函数不是线性的，我们需要使用另一种方法
    # 这里我们使用约束法，将一个目标转化为约束
    
    # 方法1: 将f2作为约束，优化f1
    # 设f2 ≥ α*f2_max + (1-α)*f2_min
    target_f2 = alpha*f2_max + (1-alpha)*f2_min
    
    # 添加f2约束: 3x1 + 3x2 ≥ target_f2
    A_new = A.copy()
    b_new = b.copy()
    A_new.append([-3, -3])  # -3x1 - 3x2 ≤ -target_f2
    b_new.append(-target_f2)
    
    # 优化f1
    res_alpha = linprog(c_f1, A_ub=A_new, b_ub=b_new, bounds=bounds, method='highs')
    
    if res_alpha.success:
        x1, x2 = res_alpha.x
        f1 = 4*x1 - 6*x2
        f2 = 3*x1 + 3*x2
        
        print(f"\n3-2 α法求解结果 (α={alpha}):")
        print(f"最优解: x1 = {x1:.2f}, x2 = {x2:.2f}")
        print(f"目标函数值:")
        print(f"f1 = {f1:.2f}")
        print(f"f2 = {f2:.2f}")
        print(f"目标f2 ≥ {target_f2:.2f} (α={alpha})")
        print("\n约束条件验证:")
        print(f"2x1 + 4x2 = {2*x1 + 4*x2:.2f} ≤ 14")
        print(f"6x1 + 3x2 = {6*x1 + 3*x2:.2f} ≤ 24")
    else:
        print(f"对于α={alpha}，优化问题无可行解")

alpha_method()