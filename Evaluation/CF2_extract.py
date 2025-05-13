# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:37:47 2024

@author: 79234
"""

from rdkit import Chem
import pandas as pd

# 读取CSV文件
data = pd.read_csv('valid_smiles_clm.csv')
smiles_list = data['SMILES']

# 用于存储符合条件的SMILES及其对应的行
filtered_rows = []

# 遍历每个SMILES表示
for index, smiles in enumerate(smiles_list):
    # 创建分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 如果分子不为空
    if mol is not None:
        # 创建一个标志用于指示是否存在符合条件的碳原子
        pfas_found = False

        # 遍历分子中的所有原子
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # 如果是碳原子
                # 获取碳原子的邻居原子
                neighbors = [neighbor.GetAtomicNum() for neighbor in atom.GetNeighbors()]

                # 检查碳原子的邻居原子是否符合条件
                if neighbors.count(9) == 2 and neighbors.count(6) == 2:
                    # 进一步检查是否为饱和脂肪碳
                    is_saturated = True
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 6:
                            # 如果邻居是碳原子，检查碳-碳键是否为单键
                            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                            if bond.GetBondType() != Chem.BondType.SINGLE:
                                is_saturated = False
                                break
                    if is_saturated:
                        pfas_found = True
                        break

        # 如果存在符合条件的碳原子
        if pfas_found:
            # 添加符合条件的SMILES及其对应行号到filtered_rows列表
            filtered_rows.append(data.iloc[index])  # 保留整行数据

# 将符合条件的SMILES输出到新的DataFrame
filtered_data = pd.DataFrame(filtered_rows)

# 将结果保存到新的CSV文件
filtered_data.to_csv('clm_CF2_Structures.csv', index=False)

print("提取的符合条件的SMILES已保存到 'Filtered_PFAS_Structures.csv'")
