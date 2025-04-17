from scipy.optimize import linprog

# 目标规划问题求解
def goal_programming():
    # 变量顺序: x1, x2, d1-, d1+, d2-, d2+, d3-, d3+, d4-, d4+
    # 共10个变量
    
    # 目标函数系数
    # min P1*d1- + P2*(4d2- + 3d3-) + P3*d4+
    # 由于scipy只能处理单层优先级，我们需要分阶段求解
    
    # 第一阶段: 最小化P1目标 (d1-)
    c_stage1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # 约束条件
    A_eq = [
        # x1 + 0.8x2 + d1- - d1+ = 112
        [1, 0.8, 1, -1, 0, 0, 0, 0, 0, 0],
        # x1 + d2- - d2+ = 80
        [1, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        # x2 + d3- - d3+ = 100
        [0, 1, 0, 0, 0, 0, 1, -1, 0, 0],
        # x1 + 0.8x2 + d4- - d4+ = 152
        [1, 0.8, 0, 0, 0, 0, 0, 0, 1, -1]
    ]
    b_eq = [112, 80, 100, 152]
    
    # 不等式约束: x1 + x2 <= 161
    A_ub = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    b_ub = [161]
    
    # 变量边界
    bounds = [
        (0, None),  # x1
        (0, None),  # x2
        (0, None),  # d1-
        (0, None),  # d1+
        (0, None),  # d2-
        (0, None),  # d2+
        (0, None),  # d3-
        (0, None),  # d3+
        (0, None),  # d4-
        (0, None)   # d4+
    ]
    
    # 第一阶段优化: 最小化d1-
    res_stage1 = linprog(c_stage1, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res_stage1.success:
        print("第一阶段优化失败")
        return
    
    # 获取第一阶段的最优d1-值
    d1_minus_opt = res_stage1.x[2]
    print(f"第一阶段结果 - 最小d1-值: {d1_minus_opt}")
    
    # 第二阶段: 在d1-最优的基础上，最小化4d2- + 3d3-
    # 添加d1- <= d1_minus_opt作为约束
    A_ub_stage2 = A_ub.copy()
    A_ub_stage2.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # d1- <= d1_minus_opt
    b_ub_stage2 = b_ub.copy()
    b_ub_stage2.append(d1_minus_opt)
    
    # 第二阶段目标函数系数: 4d2- + 3d3-
    c_stage2 = [0, 0, 0, 0, 4, 0, 3, 0, 0, 0]
    
    res_stage2 = linprog(c_stage2, A_ub=A_ub_stage2, b_ub=b_ub_stage2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res_stage2.success:
        print("第二阶段优化失败")
        return
    
    # 获取第二阶段的最优d2-和d3-值
    d2_minus_opt = res_stage2.x[4]
    d3_minus_opt = res_stage2.x[6]
    print(f"第二阶段结果 - 4d2- + 3d3-最小值: {4*d2_minus_opt + 3*d3_minus_opt}")
    print(f"d2-值: {d2_minus_opt}, d3-值: {d3_minus_opt}")
    
    # 第三阶段: 在前两阶段最优的基础上，最小化d4+
    # 添加d1- <= d1_minus_opt和4d2- + 3d3- <= 最优值作为约束
    A_ub_stage3 = A_ub_stage2.copy()
    # 添加4d2- + 3d3- <= 最优值的约束
    A_ub_stage3.append([0, 0, 0, 0, 4, 0, 3, 0, 0, 0])
    b_ub_stage3 = b_ub_stage2.copy()
    b_ub_stage3.append(4*d2_minus_opt + 3*d3_minus_opt)
    
    # 第三阶段目标函数系数: d4+
    c_stage3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    res_stage3 = linprog(c_stage3, A_ub=A_ub_stage3, b_ub=b_ub_stage3, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res_stage3.success:
        print("第三阶段优化失败")
        return
    
    # 最终结果
    x1, x2 = res_stage3.x[0], res_stage3.x[1]
    d1_minus, d1_plus = res_stage3.x[2], res_stage3.x[3]
    d2_minus, d2_plus = res_stage3.x[4], res_stage3.x[5]
    d3_minus, d3_plus = res_stage3.x[6], res_stage3.x[7]
    d4_minus, d4_plus = res_stage3.x[8], res_stage3.x[9]
    
    print("\n最终优化结果:")
    print(f"决策变量: x1 = {x1:.2f}, x2 = {x2:.2f}")
    print("\n偏差变量:")
    print(f"d1- = {d1_minus:.2f}, d1+ = {d1_plus:.2f}")
    print(f"d2- = {d2_minus:.2f}, d2+ = {d2_plus:.2f}")
    print(f"d3- = {d3_minus:.2f}, d3+ = {d3_plus:.2f}")
    print(f"d4- = {d4_minus:.2f}, d4+ = {d4_plus:.2f}")
    
    print("\n约束条件验证:")
    print(f"x1 + x2 = {x1 + x2:.2f} ≤ 161")
    print(f"x1 + 0.8x2 + d1- - d1+ = {x1 + 0.8*x2 + d1_minus - d1_plus:.2f} = 112")
    print(f"x1 + d2- - d2+ = {x1 + d2_minus - d2_plus:.2f} = 80")
    print(f"x2 + d3- - d3+ = {x2 + d3_minus - d3_plus:.2f} = 100")
    print(f"x1 + 0.8x2 + d4- - d4+ = {x1 + 0.8*x2 + d4_minus - d4_plus:.2f} = 152")
    
    print("\n目标函数值:")
    print(f"P1*d1- = {d1_minus:.2f}")
    print(f"P2*(4d2- + 3d3-) = {4*d2_minus + 3*d3_minus:.2f}")
    print(f"P3*d4+ = {d4_plus:.2f}")

goal_programming()