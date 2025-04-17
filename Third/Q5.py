import numpy as np
from collections import defaultdict

# 定义候选人
candidates = ['a', 'b', 'c', 'd']
num_candidates = len(candidates)

# 定义偏好序
preferences = [
    (8, ['a', 'b', 'c', 'd']),
    (4, ['b', 'c', 'd', 'a']),
    (6, ['b', 'd', 'a', 'c']),
    (5, ['c', 'd', 'a', 'b']),
    (5, ['d', 'a', 'c', 'b']),
    (2, ['d', 'c', 'b', 'a'])
]

# 初始化成对比较矩阵
pairwise = np.zeros((num_candidates, num_candidates), dtype=int)

# 填充成对比较矩阵
for count, pref in preferences:
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            winner = pref[i]
            loser = pref[j]
            winner_idx = candidates.index(winner)
            loser_idx = candidates.index(loser)
            pairwise[winner_idx, loser_idx] += count

# 计算Condorcet函数值
condorcet_scores = {}
for i in range(num_candidates):
    condorcet_scores[candidates[i]] = np.sum(pairwise[i, :])

# 计算Borda函数值
borda_scores = defaultdict(int)
for count, pref in preferences:
    for pos, candidate in enumerate(pref):
        borda_scores[candidate] += count * (num_candidates - 1 - pos)

# 计算Copeland函数值
copeland_scores = defaultdict(int)
for i in range(num_candidates):
    wins = 0
    losses = 0
    for j in range(num_candidates):
        if i != j:
            if pairwise[i, j] > pairwise[j, i]:
                wins += 1
            elif pairwise[i, j] < pairwise[j, i]:
                losses += 1
    copeland_scores[candidates[i]] = wins - losses

# 输出结果
print("成对比较矩阵:")
print(pairwise)

print("\nCondorcet函数值:")
for candidate in candidates:
    print(f"{candidate}: {condorcet_scores[candidate]}")

print("\nBorda函数值:")
for candidate in candidates:
    print(f"{candidate}: {borda_scores[candidate]}")

print("\nCopeland函数值:")
for candidate in candidates:
    print(f"{candidate}: {copeland_scores[candidate]}")

# 确定排序
def get_ranking(scores):
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

condorcet_ranking = get_ranking(condorcet_scores)
borda_ranking = get_ranking(borda_scores)
copeland_ranking = get_ranking(copeland_scores)

print("\nCondorcet排序:")
print(condorcet_ranking)

print("\nBorda排序:")
print(borda_ranking)

print("\nCopeland排序:")
print(copeland_ranking)