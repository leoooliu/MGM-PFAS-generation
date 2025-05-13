# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:12:11 2024

@author: 79234
"""

import pandas as pd
from rdkit import Chem


# 读取CSV文件
csv_file_path = 'NPS_smiles_frequency.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file_path)

# 定义一个函数来计算精确质量数和分子式
def calculate_mass_and_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mass = Chem.rdMolDescriptors.CalcExactMolWt(mol)
       
    
       # 计算分子式
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        return mass, formula
    else:
        return None, None

# 将计算结果添加到DataFrame中
df['exactmass'], df['formula'] = zip(*df['SMILES'].map(calculate_mass_and_formula))

# 保存带有计算结果的CSV文件
output_csv = 'nps_exactmass.csv'  # 替换为输出CSV文件路径
df.to_csv(output_csv, index=False)

print(f"已计算精确质量数和分子式，并将结果保存到 {output_csv} 文件中。")