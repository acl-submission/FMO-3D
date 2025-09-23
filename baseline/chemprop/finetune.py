import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold

from rdkit import Chem


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ========== Construction of molecular diagrams ==========
def mol_to_graph(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ])
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    n_atoms = mol.GetNumAtoms()
    adjacency = torch.zeros((n_atoms, n_atoms), dtype=torch.float)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return atom_features, adjacency

# ========== dataset ==========
class MoleculeDataset(Dataset):
    def __init__(self, df, label_columns):
        self.df = df.reset_index(drop=True)
        self.smiles = self.df["nonStereoSMILES"].tolist()
        self.file_paths = self.df["file_path"].astype(str).tolist() if "file_path" in self.df.columns else [""] * len(self.df)
        self.labels = self.df[label_columns].values.astype(np.float32) if len(label_columns) > 0 else None
        self.label_columns = label_columns

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        atom_features, adjacency = mol_to_graph(mol)
        label = torch.tensor(self.labels[idx]) if self.labels is not None else None
        file_path = self.file_paths[idx]
        return atom_features, adjacency, label, smi, file_path

def collate_fn(batch):
    atom_features, adjacencies, labels, smiles, file_paths = zip(*batch)
    labels = None if labels[0] is None else torch.stack(labels)
    return list(atom_features), list(adjacencies), labels, list(smiles), list(file_paths)

# ========== Chemprop Style model ==========
class MPNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.res = (in_dim == out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h, adj):
        m = torch.matmul(adj, h)
        h_new = self.linear(h + m)
        if self.res:
            h_new = h_new + h
        return self.norm(F.relu(h_new))

class ChempropLikeMPNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, output_dim,
                 aggregation="mean", agg_norm=100, dropout=0.0, ffn_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MPNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.aggregation = aggregation
        self.agg_norm = agg_norm

        # FFN
        ffn = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(ffn_layers - 1):
            ffn.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        ffn.append(nn.Linear(hidden_dim, output_dim))
        self.readout = nn.Sequential(*ffn)

        if aggregation == "attentive":
            self.att = nn.Linear(hidden_dim, 1)

    def _aggregate(self, h):
        if self.aggregation == "sum":
            g = torch.sum(h, dim=0)
        elif self.aggregation == "mean":
            g = torch.mean(h, dim=0)
        elif self.aggregation == "norm":
            g = torch.sum(h, dim=0) / self.agg_norm
        elif self.aggregation == "attentive":
            att_weights = torch.softmax(self.att(h), dim=0)
            g = torch.sum(att_weights * h, dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        return g

    def encode(self, atom_features, adjacency):
        # Return the graph-level embedding (before reading)
        h = atom_features
        for layer in self.layers:
            h = layer(h, adjacency)
        g = self._aggregate(h)
        g = self.dropout(g)
        return g  # [hidden_dim]

    def forward(self, atom_features, adjacency):
        g = self.encode(atom_features, adjacency)
        return self.readout(g)  # logits

# ========== Training and Evaluation ==========
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for atom_features_list, adjacencies, labels, _, _ in loader:
        if labels is None:
            continue
        batch_outputs, batch_labels = [], []
        for af, adj, label in zip(atom_features_list, adjacencies, labels):
            af, adj, label = af.to(device), adj.to(device), label.to(device)
            out = model(af, adj)
            batch_outputs.append(out)
            batch_labels.append(label)
        outputs = torch.stack(batch_outputs)
        batch_labels = torch.stack(batch_labels)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for atom_features_list, adjacencies, labels, _, _ in loader:
            if labels is None:
                continue
            batch_outputs, batch_labels = [], []
            for af, adj, label in zip(atom_features_list, adjacencies, labels):
                af, adj, label = af.to(device), adj.to(device), label.to(device)
                out = model(af, adj)
                batch_outputs.append(out)
                batch_labels.append(label)
            outputs = torch.stack(batch_outputs)
            batch_labels = torch.stack(batch_labels)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            all_labels.append(batch_labels.cpu())
            all_outputs.append(outputs.cpu())
    if len(all_outputs) == 0:
        return {"loss": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}
    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)
    preds = (torch.sigmoid(all_outputs) > 0.5).int()
    return {
        "loss": total_loss / len(loader),
        "precision": precision_score(all_labels, preds, average="micro", zero_division=0),
        "recall": recall_score(all_labels, preds, average="micro", zero_division=0),
        "f1": f1_score(all_labels, preds, average="micro", zero_division=0),
        "auc": roc_auc_score(all_labels, torch.sigmoid(all_outputs), average="micro")
    }

# ========== Reasoning: Prediction and Coding ==========
@torch.no_grad()
def predict_and_encode(model, loader, device):
    model.eval()
    all_logits, all_embeddings = [], []
    for atom_features_list, adjacencies, _, _, _ in loader:
        for af, adj in zip(atom_features_list, adjacencies):
            af, adj = af.to(device), adj.to(device)
            g = model.encode(af, adj)
            logits = model.readout(g)
            all_embeddings.append(g.cpu().unsqueeze(0))
            all_logits.append(logits.cpu().unsqueeze(0))
    if len(all_logits) == 0:
        return None, None
    return torch.cat(all_logits, dim=0), torch.cat(all_embeddings, dim=0)

# ========== HDF5 Write ==========
def write_h5(output_path, df, label_cols, embeddings, file_paths, smiles):
    labels_values = df[label_cols].astype(str).values.tolist() if len(label_cols) > 0 else []
    labels_with_header = [label_cols] + labels_values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        # 1) file_path
        file_path_arr = np.array([str(x) for x in file_paths], dtype=object)
        f.create_dataset("file_path", data=file_path_arr, dtype=h5py.string_dtype(encoding="utf-8"))

        # 2) nonStereoSMILES
        smiles_arr = np.array([str(x) for x in smiles], dtype=object)
        f.create_dataset("nonStereoSMILES", data=smiles_arr, dtype=h5py.string_dtype(encoding="utf-8"))

        # 3) embeddings
        if embeddings is None:
            emb = np.random.rand(len(df), 300).astype("float32")
        else:
            emb = embeddings.astype("float32")
        f.create_dataset("embeddings", data=emb)

        n_rows = len(labels_with_header)
        n_cols = len(label_cols)
        ds = f.create_dataset("labels", shape=(n_rows, n_cols), dtype=h5py.string_dtype(encoding="utf-8"))
        for i, row in enumerate(labels_with_header):
            ds[i] = [str(x) for x in row]

    print(f"✅ Has been written：{output_path}")

# ========== Main process ==========
def train_and_dump(
    train_path,
    test_path,
    out_dir=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    K=2,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "nonStereoSMILES" not in train_df.columns:
        raise ValueError("The training set is missing the nonStereoSMILES column")
    if "nonStereoSMILES" not in test_df.columns:
        raise ValueError("The test set is missing the nonStereoSMILES column")
    if "file_path" not in train_df.columns:
        train_df["file_path"] = ""
    if "file_path" not in test_df.columns:
        test_df["file_path"] = ""

    skip_cols = {"file_path", "nonStereoSMILES", "descriptors"}
    label_cols = [c for c in train_df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    print(f"Detected {len(label_cols)} A column of numerical labels: {label_cols}")

    # config
    HIDDEN_SIZE = 300
    DEPTH = 3
    DROPOUT = 0.0
    BATCH_SIZE = 50
    MAX_EPOCH = 200

    # Dataset & K fold
    dataset = MoleculeDataset(train_df, label_cols)
    kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

    # Training
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n==== Fold {fold + 1}/{K} ====")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = ChempropLikeMPNN(
            input_dim=6,
            hidden_dim=HIDDEN_SIZE,
            depth=DEPTH,
            output_dim=len(label_cols),
            aggregation="mean",
            dropout=DROPOUT,
            ffn_layers=2
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        best_auc, patience, wait = 0.0, 10, 0
        for epoch in range(1, MAX_EPOCH + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = eval_epoch(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch:03d} | LR {0.001:.6f} | Train Loss {train_loss:.4f} | Val AUC {val_metrics['auc']:.4f}")

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                wait = 0
                torch.save(model.state_dict(), f"best_model_fold{fold}.pt")
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered")
                    break

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                wait = 0
                torch.save(model.state_dict(), f"best_model_fold{fold}.pt")
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered")
                    break

    print("\n=== Reasoning (Complete Set of train & test) ===")
    full_train_ds = MoleculeDataset(train_df, label_cols)
    full_test_ds = MoleculeDataset(test_df, label_cols)  # 即使 test 没有标签列也兼容
    full_train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    full_test_loader = DataLoader(full_test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_train_logits, all_train_embs = [], []
    all_test_logits, all_test_embs = [], []

    for fold in range(K):
        model = ChempropLikeMPNN(
            input_dim=6,
            hidden_dim=HIDDEN_SIZE,
            depth=DEPTH,
            output_dim=len(label_cols),
            aggregation="mean",
            dropout=DROPOUT,
            ffn_layers=2
        ).to(device)
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pt", map_location=device))

        tr_logits, tr_emb = predict_and_encode(model, full_train_loader, device)
        te_logits, te_emb = predict_and_encode(model, full_test_loader, device)

        all_train_logits.append(torch.sigmoid(tr_logits))
        all_train_embs.append(tr_emb)
        all_test_logits.append(torch.sigmoid(te_logits))
        all_test_embs.append(te_emb)

    # Integrated average
    mean_train_logits = torch.mean(torch.stack(all_train_logits, dim=0), dim=0).numpy()
    mean_test_logits  = torch.mean(torch.stack(all_test_logits,  dim=0), dim=0).numpy()
    mean_train_embs   = torch.mean(torch.stack(all_train_embs,   dim=0), dim=0).numpy()
    mean_test_embs    = torch.mean(torch.stack(all_test_embs,    dim=0), dim=0).numpy()

    # ====== Write the prediction results of test back to CSV (optional, consistent with your original logic) ======
    test_with_pred = test_df.copy()
    for i, col in enumerate(label_cols):
        test_with_pred[f"pred_prob_{col}"] = mean_test_logits[:, i]
        test_with_pred[f"pred_label_{col}"] = (mean_test_logits[:, i] > 0.5).astype(int)

    base, ext = os.path.splitext(test_path)
    new_test_path = f"{base}_with_pred{ext}"
    test_with_pred.to_csv(new_test_path, index=False, encoding="utf-8-sig")
    print(f"✅ The test set prediction results have been saved to {new_test_path}")

    # ====== HDF5（train & test），included file_path / nonStereoSMILES / embeddings / labels ======
    out_dir = out_dir or os.path.dirname(train_path) or "."
    train_h5 = os.path.join(out_dir, "train_embeddings(s).h5")
    test_h5  = os.path.join(out_dir, "test_embeddings(s).h5")

    write_h5(
        train_h5,
        df=train_df,
        label_cols=[c for c in train_df.columns if c not in {"file_path", "nonStereoSMILES", "descriptors"}],
        embeddings=mean_train_embs,
        file_paths=train_df["file_path"].astype(str).tolist() if "file_path" in train_df.columns else [""] * len(train_df),
        smiles=train_df["nonStereoSMILES"].astype(str).tolist(),
    )

    write_h5(
        test_h5,
        df=test_df,
        label_cols=[c for c in test_df.columns if c not in {"file_path", "nonStereoSMILES", "descriptors"}],
        embeddings=mean_test_embs,
        file_paths=test_df["file_path"].astype(str).tolist() if "file_path" in test_df.columns else [""] * len(test_df),
        smiles=test_df["nonStereoSMILES"].astype(str).tolist(),
    )

    print("🎉 The entire process is completed！")

# ========= Operation entry =========
if __name__ == "__main__":
    # —— Change the following two lines to your actual path ——
    train_path = r"C:\Users\AA\Desktop\datasets\openpom_train_dataset_secondary(7).csv"
    test_path = r"C:\Users\AA\Desktop\datasets\openpom_test_dataset_secondary(7).csv"

    out_dir = None

    train_and_dump(train_path, test_path, out_dir)
