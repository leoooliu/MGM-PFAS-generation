# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 04:43:43 2024

@author: 79234
"""

from collections import Counter
import csv
import pandas as pd
data = pd.read_csv('CLM_PFAS_train_excluded.csv')
smiles_list =data['SMILES']
# 统计SMILES的频率
smiles_frequency = Counter(smiles_list)

# 提取唯一的SMILES和它们的频率
unique_smiles = list(smiles_frequency.keys())
frequencies = list(smiles_frequency.values())

# 创建一个包含唯一SMILES和频率的DataFrame
import pandas as pd
data = {'SMILES': unique_smiles, '频率': frequencies}
df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件
output_csv = 'CLM_smiles_frequency.csv'  # 替换为输出CSV文件的路径
df.to_csv(output_csv, index=False)

print(f"已统计SMILES的频率并将结果保存到 {output_csv} 文件中。")