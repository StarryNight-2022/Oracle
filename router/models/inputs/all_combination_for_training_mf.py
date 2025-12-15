import random
from itertools import combinations

# 原始列表
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

all_combinations = []

for k in range(2, 10):
    #使用itertools.combinations生成所有组合，然后随机选择
    combinations_for_k = list(combinations(my_list, k))
    all_combinations.extend(combinations_for_k)
print(len(all_combinations))