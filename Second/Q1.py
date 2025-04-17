import numpy as np
import pandas as pd

# 原始数据
data = {
    '人均专著': [0.1, 0.2, 0.6, 0.3, 2.8],
    '生师比': [5, 7, 10, 4, 2],
    '科研经费': [5000, 4000, 1260, 3000, 284],
    '逾期毕业率': [4.7, 2.2, 3.0, 3.9, 1.2]
}

df = pd.DataFrame(data)

# 定义规范化参数
f0 = 2  # 下限临界值
f1 = 6  # 最优区间下限
f2 = 7  # 最优区间上限
f_optimal = 12  # 上限临界值

# 1. 极差变换法规范化
def min_max_normalize(series, benefit=True):
    if benefit:
        return (series - series.min()) / (series.max() - series.min())
    else:
        return (series.max() - series) / (series.max() - series.min())

# 对第1列(人均专著)-效益型
df['人均专著_极差'] = min_max_normalize(df['人均专著'], benefit=True)

# 对第3列(科研经费)-效益型
df['科研经费_极差'] = min_max_normalize(df['科研经费'], benefit=True)

# 对第4列(逾期毕业率)-成本型
df['逾期毕业率_极差'] = min_max_normalize(df['逾期毕业率'], benefit=False)

# 2. 向量归一化方法
def vector_normalize(series, benefit=True):
    norm = np.sqrt((series**2).sum())
    if benefit:
        return series / norm
    else:
        return 1 - (series / norm)

df['人均专著_向量'] = vector_normalize(df['人均专著'], benefit=True)
df['科研经费_向量'] = vector_normalize(df['科研经费'], benefit=True)
df['逾期毕业率_向量'] = vector_normalize(df['逾期毕业率'], benefit=False)

# 3. 修正后的中性属性规范化方法(对第2列-生师比)
def neutral_normalize_corrected(series, f0, f1, f2, f_optimal):
    normalized = []
    for x in series:
        if x < f0:
            normalized.append(0)  
        elif x < f1:
            normalized.append(1 - (f1 - x) / (f1 - f0))  # 线性增长到1
        elif x <= f2:  # 最优区间，直接取1
            normalized.append(1.0)
        elif x < f_optimal:
            normalized.append(1 - (x - f2) / (f_optimal - f2))  # 线性下降到0
        else:
            normalized.append(0)  
    return pd.Series(normalized)

df['生师比_中性'] = neutral_normalize_corrected(df['生师比'], f0, f1, f2, f_optimal)

# 显示结果
print("规范化结果：")
print(df[['人均专著_极差', '人均专著_向量', 
           '生师比_中性', 
           '科研经费_极差', '科研经费_向量', 
           '逾期毕业率_极差', '逾期毕业率_向量']])