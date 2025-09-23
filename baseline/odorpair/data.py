# from pairing.data import PairData
# import pairing.data
# import torch_geometric as pyg
# import tqdm
# import torch
# import os
#
#
# out_dir = "single"
# train_fname = "trainsingles.pt"
# test_fname = "testsingles.pt"
#
# # With very clever dictionary comprehension this could be easier.
# def get_data_s(singles,all_notes,data):
#     notes_s = singles[data.smiles_s]
#     y = pairing.data.multi_hot(singles[data.smiles_s],all_notes)
#     data_s = pyg.data.Data(x=data.x_s,edge_index=data.edge_index_s,edge_attr=data.edge_attr_s,y=y.float(),smiles=data.smiles_s)
#     return data_s
#
# def get_data_t(singles,all_notes,data):
#     notes_t = singles[data.smiles_t]
#     y = pairing.data.multi_hot(singles[data.smiles_t],all_notes)
#     data_t = pyg.data.Data(x=data.x_t,edge_index=data.edge_index_t,edge_attr=data.edge_attr_t,y=y.float(),smiles=data.smiles_t)
#     return data_t
#
# # This is a complicated function, but essentially
# # the goal is to ensure train/test separation along the same lines
# # as the blended pair dataset. As a result, we go through the blended
# # pair dataset, and get the individual notes for each molecule across
# # every pair, separated by train and test.
# def build_dataset(is_train):
#     pair_data = pairing.data.Dataset(is_train=is_train)
#     singles = pairing.data.get_singles()
#     all_notes = pairing.data.get_all_notes()
#
#     all_data = []
#     seen = set()
#
#     for d in tqdm.tqdm(pair_data):
#         if not d.smiles_s in seen:
#             try:
#                 data_s = get_data_s(singles,all_notes,d)
#                 all_data.append(data_s)
#                 seen.add(data_s.smiles)
#             except AttributeError:
#                 pass
#
#         if not d.smiles_t in seen:
#             try:
#                 data_t = get_data_t(singles,all_notes,d)
#                 all_data.append(data_t)
#                 seen.add(data_t.smiles)
#             except AttributeError:
#                 pass
#
#     return all_data
#
# def save(data_list, fname):
#     data, slices = pyg.data.InMemoryDataset.collate(data_list)
#     torch.save((data, slices), os.path.join(out_dir, fname))
#
# def build():
#     all_notes = pairing.data.get_all_notes()
#
#     train_data_list = build_dataset(True)
#     test_data_list = build_dataset(False)
#
#     print(f"Built train dataset of len = {len(train_data_list)}")
#     save(train_data_list,train_fname)
#
#     print(f"Built test dataset of len = {len(test_data_list)}")
#     save(test_data_list,test_fname)
#
# # TODO: Refactor this into the pairing.data.Dataset
# class Dataset(pyg.data.InMemoryDataset):
#     def __init__(self, is_train):
#         super().__init__(out_dir)
#         if is_train:
#             self.data, self.slices = torch.load(
#                 os.path.join(out_dir, train_fname))
#         else:
#             self.data, self.slices = torch.load(
#                 os.path.join(out_dir, test_fname))
#
#     @classmethod
#     def num_classes(cls):
#         return 60
#
#     @classmethod
#     def num_node_features(cls):
#         return pairing.data.Dataset.num_node_features()
#
#     @classmethod
#     def num_edge_features(cls):
#         return pairing.data.Dataset.num_edge_features()
#
# if __name__ == "__main__":
#     build()

# import pandas as pd
# import torch
# import torch_geometric as pyg
# from torch_geometric.data import InMemoryDataset, Data
# from torch_geometric.loader import DataLoader  # 导入 DataLoader
# from rdkit import Chem
# import os
# import tqdm
#
# out_dir = "single"
# train_fname = "trainsingles.pt"
# test_fname = "testsingles.pt"
#
# def smiles_to_graph(smiles):
#     """将 SMILES 转换为 PyTorch Geometric 的 Data 对象"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#
#     # 提取节点特征（与原始 num_node_features=9 一致）
#     node_feat = []
#     for atom in mol.GetAtoms():
#         features = [
#             atom.GetAtomicNum(),  # 原子序数
#             atom.GetDegree(),
#             atom.GetFormalCharge(),
#             atom.GetNumRadicalElectrons(),
#             atom.GetIsAromatic(),
#             atom.GetHybridization().real,
#             atom.GetTotalNumHs(),
#             atom.IsInRing(),
#             0  # 占位符
#         ]
#         node_feat.append(features)
#
#     # 提取边和边特征（与 num_edge_features=3 一致）
#     edge_index = []
#     edge_attr = []
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
#         edge_index.append([i, j])
#         edge_index.append([j, i])  # 无向图
#         bond_type = bond.GetBondTypeAsDouble()
#         edge_attr.append([bond_type, int(bond.IsInRing()), 0])
#         edge_attr.append([bond_type, int(bond.IsInRing()), 0])
#
#     # 转换为张量
#     x = torch.tensor(node_feat, dtype=torch.float)
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
#
# def build_dataset(csv_file):
#     """从 CSV 文件构建单分子数据集"""
#     data = pd.read_csv(csv_file)
#     labels_cols = ['animal aroma', 'fermentation aroma', 'floral', 'fruity',
#                    'herbal', 'maillard', 'milky', 'special aroma', 'sweet aroma']
#     smiles = data['nonStereoSMILES']
#     labels = data[labels_cols].values
#
#     data_list = []
#     for smi, label in tqdm.tqdm(zip(smiles, labels), total=len(smiles)):
#         graph = smiles_to_graph(smi)
#         if graph is not None:
#             graph.y = torch.tensor(label, dtype=torch.float)
#             data_list.append(graph)
#
#     return data_list
#
# def save(data_list, fname):
#     data, slices = InMemoryDataset.collate(data_list)
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save((data, slices), os.path.join(out_dir, fname))
#
# def build():
#     """构建训练和测试数据集"""
#     train_csv = "openpom_train_dataset_primary.csv"  # 训练集路径
#     test_csv = "openpom_test_dataset_primary.csv"    # 测试集路径
#
#     train_data_list = build_dataset(train_csv)
#     test_data_list = build_dataset(test_csv)
#
#     print(f"Built train dataset of len = {len(train_data_list)}")
#     save(train_data_list, train_fname)
#     print(f"Built test dataset of len = {len(test_data_list)}")
#     save(test_data_list, test_fname)
# def loader(dataset, batch_size):
#     """为单分子数据集创建 DataLoader"""
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)
# class Dataset(InMemoryDataset):
#     def __init__(self, is_train):
#         super().__init__(out_dir)
#         if is_train:
#             self.data, self.slices = torch.load(os.path.join(out_dir, train_fname))
#         else:
#             self.data, self.slices = torch.load(os.path.join(out_dir, test_fname))
#
#     @classmethod
#     def num_classes(cls):
#         return 9  # 你的标签数量
#
#     @classmethod
#     def num_node_features(cls):
#         return 9  # 与原始一致
#
#     @classmethod
#     def num_edge_features(cls):
#         return 3  # 与原始一致
#
# if __name__ == "__main__":
#     build()

# import pandas as pd
# import torch
# import torch_geometric as pyg
# from torch_geometric.data import InMemoryDataset, Data
# from torch_geometric.loader import DataLoader  # 导入 DataLoader
# from rdkit import Chem
# import os
# import tqdm
#
# out_dir = "single_secondary"  # 单独存放 secondary 数据
# train_fname = "trainsingles.pt"
# test_fname = "testsingles.pt"
#
# def smiles_to_graph(smiles):
#     """将 SMILES 转换为 PyTorch Geometric 的 Data 对象"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#
#     # 提取节点特征（与原始 num_node_features=9 一致）
#     node_feat = []
#     for atom in mol.GetAtoms():
#         features = [
#             atom.GetAtomicNum(),  # 原子序数
#             atom.GetDegree(),
#             atom.GetFormalCharge(),
#             atom.GetNumRadicalElectrons(),
#             atom.GetIsAromatic(),
#             atom.GetHybridization().real,
#             atom.GetTotalNumHs(),
#             atom.IsInRing(),
#             0  # 占位符
#         ]
#         node_feat.append(features)
#
#     # 提取边和边特征（与 num_edge_features=3 一致）
#     edge_index = []
#     edge_attr = []
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
#         edge_index.append([i, j])
#         edge_index.append([j, i])  # 无向图
#         bond_type = bond.GetBondTypeAsDouble()
#         edge_attr.append([bond_type, int(bond.IsInRing()), 0])
#         edge_attr.append([bond_type, int(bond.IsInRing()), 0])
#
#     # 转换为张量
#     x = torch.tensor(node_feat, dtype=torch.float)
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
#
# def build_dataset(csv_file):
#     """从 CSV 文件构建单分子数据集"""
#     data = pd.read_csv(csv_file)
#
#     # secondary 数据集的标签字段
#     labels_cols = [
#         'beany','berry','caramel sweet','cheesy aroma','chemical notes',
#         'citrus','complex floral','complex fruity','complex milky','complex yeast aroma',
#         'cooked','creamy aroma','ethanolic aroma','fatty','fermented sweet',
#         'fresh floral','grassy','honey sweet','light floral','meaty',
#         'medicinal','melon','mineral','musky','nutty',
#         'pome','roasted','sensory','spicy','stone fruit',
#         'subtle floral','sweet floral','vegetal','woody'
#     ]
#
#     smiles = data['nonStereoSMILES']
#     labels = data[labels_cols].values
#
#     data_list = []
#     for smi, label in tqdm.tqdm(zip(smiles, labels), total=len(smiles)):
#         graph = smiles_to_graph(smi)
#         if graph is not None:
#             graph.y = torch.tensor(label, dtype=torch.float)
#             data_list.append(graph)
#
#     return data_list
#
# def save(data_list, fname):
#     data, slices = InMemoryDataset.collate(data_list)
#     os.makedirs(out_dir, exist_ok=True)
#     torch.save((data, slices), os.path.join(out_dir, fname))
#
# def build():
#     """构建训练和测试数据集"""
#     train_csv = "openpom_train_dataset_secondary.csv"  # secondary 训练集路径
#     test_csv = "openpom_test_dataset_secondary.csv"    # secondary 测试集路径
#
#     train_data_list = build_dataset(train_csv)
#     test_data_list = build_dataset(test_csv)
#
#     print(f"Built train dataset of len = {len(train_data_list)}")
#     save(train_data_list, train_fname)
#     print(f"Built test dataset of len = {len(test_data_list)}")
#     save(test_data_list, test_fname)
#
# def loader(dataset, batch_size):
#     """为单分子数据集创建 DataLoader"""
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# class Dataset(InMemoryDataset):
#     def __init__(self, is_train):
#         super().__init__(out_dir)
#         if is_train:
#             self.data, self.slices = torch.load(os.path.join(out_dir, train_fname))
#         else:
#             self.data, self.slices = torch.load(os.path.join(out_dir, test_fname))
#
#     @classmethod
#     def num_classes(cls):
#         return 34  # secondary 数据集标签数
#
#     @classmethod
#     def num_node_features(cls):
#         return 9  # 与原始一致
#
#     @classmethod
#     def num_edge_features(cls):
#         return 3  # 与原始一致
#
# if __name__ == "__main__":
#     build()


import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import os
import tqdm

# out_dir = "single"  # Store the data of new large-scale tags
out_dir = "single_secondary"
train_fname = "trainsingles.pt"
test_fname = "testsingles.pt"

def smiles_to_graph(smiles):
    """Convert SMILES to the Data object of PyTorch Geometric"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (fixed 9-dimensional)
    node_feat = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
            atom.GetHybridization().real,
            atom.GetTotalNumHs(),
            atom.IsInRing(),
            0
        ]
        node_feat.append(features)

    # Edge features (fixed 3D)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        bond_type = bond.GetBondTypeAsDouble()
        edge_attr.append([bond_type, int(bond.IsInRing()), 0])
        edge_attr.append([bond_type, int(bond.IsInRing()), 0])

    x = torch.tensor(node_feat, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

# def build_dataset(csv_file):
#     """从 CSV 文件构建数据集（自动识别标签列）"""
#     # data = pd.read_csv(csv_file, sep="\t")  # 注意分隔符，有些文件是制表符分隔
#     data = pd.read_csv(csv_file, sep=None, engine="python")
#     print(data.head())
#     print(data.columns.tolist())
#
#     # 自动识别标签列（排除 SMILES 和 descriptors）
#     labels_cols = [col for col in data.columns if col not in ["nonStereoSMILES", "descriptors"]]
#
#     smiles = data["nonStereoSMILES"]
#     labels = data[labels_cols].values
#
#     data_list = []
#     for smi, label in tqdm.tqdm(zip(smiles, labels), total=len(smiles)):
#         graph = smiles_to_graph(smi)
#         if graph is not None:
#             graph.y = torch.tensor(label, dtype=torch.float)
#             data_list.append(graph)
#
#     return data_list, labels_cols
def build_dataset(csv_file):
    """Build a dataset from a CSV file (automatically identify label columns)"""
    data = pd.read_csv(csv_file, sep=None, engine="python")
    print(data.head())
    print(data.columns.tolist())

    # Tag columns = Columns other than file_path, nonStereoSMILES, and descriptors
    labels_cols = [col for col in data.columns if col not in ["file_path", "nonStereoSMILES", "descriptors"]]

    file_paths = data["file_path"].values
    smiles = data["nonStereoSMILES"].values
    labels = data[labels_cols].values

    data_list = []
    for smi, label, fpath in tqdm.tqdm(zip(smiles, labels, file_paths), total=len(smiles)):
        graph = smiles_to_graph(smi)
        if graph is not None:
            graph.y = torch.tensor(label, dtype=torch.float)
            graph.file_path = fpath   # ⭐ Save the file_path information
            graph.smiles = smi        # ⭐ Save the original SMILES (optional, already available)
            data_list.append(graph)

    return data_list, labels_cols

def save(data_list, fname):
    data, slices = InMemoryDataset.collate(data_list)
    os.makedirs(out_dir, exist_ok=True)
    torch.save((data, slices), os.path.join(out_dir, fname))

def build():
    train_csv = "openpom_train_dataset_secondary.csv"  # Your new training set path
    test_csv = "openpom_test_dataset_secondary.csv"    # The path of your new test set

    train_data_list, labels_cols = build_dataset(train_csv)
    test_data_list, _ = build_dataset(test_csv)

    print(f"Number of labels: {len(labels_cols)}")
    print(f"Built train dataset of len = {len(train_data_list)}")
    save(train_data_list, train_fname)
    print(f"Built test dataset of len = {len(test_data_list)}")
    save(test_data_list, test_fname)

    # Save the label column for future use
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for l in labels_cols:
            f.write(l + "\n")

def loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Dataset(InMemoryDataset):
    def __init__(self, is_train):
        super().__init__(out_dir)
        if is_train:
            self.data, self.slices = torch.load(os.path.join(out_dir, train_fname))
        else:
            self.data, self.slices = torch.load(os.path.join(out_dir, test_fname))

    @classmethod
    def num_classes(cls):
        labels_file = os.path.join(out_dir, "labels.txt")
        with open(labels_file, "r", encoding="utf-8") as f:
            return len(f.readlines())

    @classmethod
    def num_node_features(cls):
        return 9

    @classmethod
    def num_edge_features(cls):
        return 3

if __name__ == "__main__":
    build()
    dataset = Dataset(is_train=True)
    print(dataset[0].file_path)  # The original file_path will be displayed
    print(dataset[0].smiles)  # It will display the original SMILES
    print(dataset[0].y)  # Only include tags
