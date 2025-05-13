from rdkit import Chem
import pandas as pd

# 读取CSV文件
data = pd.read_csv('valid_smiles_clm.csv')
smiles_list = data['SMILES']

# 用于存储符合条件的整行数据
filtered_rows = []

# 遍历每个SMILES表示
for index, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        # 创建一个字典用于存储每个碳原子的邻居类型和数量
        carbon_neighbors = {}

        # 遍历分子中的所有原子
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # 如果是碳原子
                carbon_neighbors[atom.GetIdx()] = {'F': 0, 'H': 0, 'Cl': 0, 'Br': 0, 'I': 0}
                
                # 遍历碳原子的邻居原子
                for neighbor in atom.GetNeighbors():
                    neighbor_symbol = neighbor.GetSymbol()
                    if neighbor_symbol in carbon_neighbors[atom.GetIdx()]:
                        carbon_neighbors[atom.GetIdx()][neighbor_symbol] += 1

        # 检查是否符合-CF3结构条件
        for carbon_idx, neighbors in carbon_neighbors.items():
            if neighbors['F'] >= 3 and sum(neighbors.values()) == 3:
                # 如果符合条件，将整个行添加到filtered_rows
                filtered_rows.append(data.iloc[index])  # 保留整行数据
                break

# 将符合条件的行输出到新的DataFrame
output_data = pd.DataFrame(filtered_rows)

# 将结果保存到新的CSV文件
output_data.to_csv('clm_CF3_Structures.csv', index=False)

print("提取的符合-CF3结构的SMILES及其所在行已保存到 'nps_CF3_Structures.csv'")