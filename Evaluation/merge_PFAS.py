# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 04:31:52 2024

@author: 79234
"""
import pandas as pd
# 读取第一个文本文件中的SMILES
file1 = 'pfas_nps_CF3.txt'  # 替换为第一个文本文件的路径
with open(file1, 'r') as f1:
    smiles1 = f1.read().splitlines()

# 读取第二个文本文件中的SMILES
file2 = 'pfas_nps_CF2.txt'  # 替换为第二个文本文件的路径
with open(file2, 'r') as f2:
    smiles2 = f2.read().splitlines()

# 合并两个SMILES列表
merged_smiles = smiles1 + smiles2
print(len(merged_smiles))
merged_smiles=pd.DataFrame(merged_smiles ,columns=['SMILES'])
merged_smiles.to_csv('clm_pfas_merged.csv')
'''
# 统计相同SMILES的频率
smiles_frequency = {}
for smiles in merged_smiles:
    if smiles in smiles_frequency:
        smiles_frequency[smiles] += 1
    else:
        smiles_frequency[smiles] = 1

# 打印相同SMILES及其频率
for smiles, frequency in smiles_frequency.items():
    print(f"SMILES: {smiles}, 频率: {frequency} 次")'''