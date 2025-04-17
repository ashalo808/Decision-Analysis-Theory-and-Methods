from scipy.optimize import linprog

# 3-1 主要目标法求解
def primary_objective_method():
    # 确定主要目标（这里选择产值最大作为主要目标）
    # 目标函数系数（linprog默认是最小化，所以取负）
    c = [-400, -600]  # max f1 = 400x1 + 600x2 → min -400x1 - 600x2
    
    # 约束条件系数矩阵
    A = [
        [4, 5],    # 4x1 + 5x2 ≤ 200
        [9, 4],    # 9x1 + 4x2 ≤ 240
        [3, 10],   # 3x1 + 10x2 ≤ 300
        [-400, -600],  # 400x1 + 600x2 ≥ 20000 (总产值约束)
        [3, 2]     # 3x1 + 2x2 ≤ 90 (污染约束)
    ]
    
    # 约束条件右侧值
    b = [200, 240, 300, -20000, 90]
    
    # 变量边界
    x_bounds = (0, None)
    y_bounds = (0, None)
    
    # 求解线性规划问题
    result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='highs')
    
    if result.success:
        x1, x2 = result.x
        f1 = 400*x1 + 600*x2
        f2 = 70*x1 + 120*x2
        f3 = 3*x1 + 2*x2
        
        print("\n3-1 主要目标法求解结果:")
        print(f"最优解: x1 = {x1:.2f}, x2 = {x2:.2f}")
        print(f"目标函数值:")
        print(f"f1(产值) = {f1:.2f} 元")
        print(f"f2(利润) = {f2:.2f} 元")
        print(f"f3(污染) = {f3:.2f} 单位")
        print("\n约束条件验证:")
        print(f"4x1 + 5x2 = {4*x1 + 5*x2:.2f} ≤ 200")
        print(f"9x1 + 4x2 = {9*x1 + 4*x2:.2f} ≤ 240")
        print(f"3x1 + 10x2 = {3*x1 + 10*x2:.2f} ≤ 300")
        print(f"总产值 = {f1:.2f} ≥ 20000")
        print(f"污染量 = {f3:.2f} ≤ 90")
    else:
        print("优化问题无可行解")

primary_objective_method()