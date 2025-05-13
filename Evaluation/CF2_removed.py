# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 04:30:19 2024

@author: 79234
"""

import pandas as pd

# 读取两个数据表
data1 = pd.read_csv('nps_ca_all_valid_smiles.csv')

input_file = 'pfas_nps_CF2.txt'  # 替换为你的文本文件路径
with open(input_file, 'r') as f:
    smiles_list = f.read().splitlines()

# 提取两个数据表中的SMILES列
smiles1 = data1['SMILES']
print(len(smiles1))
smiles2 = smiles_list

# 将SMILES列转换为集合，以便进行快速查找
smiles_set2 = set(smiles2)

# 剔除data1中与data2重复的SMILES
filtered_data1 = data1[~smiles1.isin(smiles_set2)]
print(len(filtered_data1))

# 将剔除后的数据保存到新的CSV文件
filtered_data1.to_csv('NPS_filtered_output.csv', index=False)

print("已剔除与data2中重复的SMILES，并保存到filtered_data1.csv文件中。")