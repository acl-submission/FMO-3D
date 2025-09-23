
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
