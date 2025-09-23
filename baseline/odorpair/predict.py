import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from single.data import Dataset
import pickle
import os
from torch_geometric.nn import GINConv, global_mean_pool

# Set up the equipment
device = "cuda" if torch.cuda.is_available() else "cpu"

# The model definition copied from main_single.py
class GCN(torch.nn.Module):
    def __init__(self, num_convs, num_linear, embedding_size, architecture):
        super(GCN, self).__init__()
        self.embedding_size = embedding_size
        self.convs = torch.nn.ModuleList()
        if architecture == "GIN":
            for i in range(num_convs):
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(embedding_size if i else Dataset.num_node_features(), embedding_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(embedding_size, embedding_size),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
        self.linear = make_sequential(num_linear, embedding_size, embedding_size)

    def forward(self, x, edge_index, edge_attr, batch_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch_index)
        return self.linear(x)

def make_sequential(num_layers, in_dim, out_dim, is_last=False):
    layers = []
    if num_layers == 0:
        return torch.nn.Identity()
    for i in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_dim if i == 0 else out_dim, out_dim))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_dim if num_layers == 1 else out_dim, out_dim))
    if not is_last:
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

class SingleMoleculePredictor(torch.nn.Module):
    def __init__(self, num_convs, num_linear, embedding_size, architecture):
        super(SingleMoleculePredictor, self).__init__()
        self.gcn = GCN(num_convs, num_linear, embedding_size, architecture)
        self.out = make_sequential(num_linear, embedding_size, Dataset.num_classes(), is_last=True)

    def forward(self, x, edge_index, edge_attr, batch_index):
        emb = self.gcn(x, edge_index, edge_attr, batch_index)
        return self.out(emb)


def collate_fn(batch):
    y = torch.stack([data.y for data in batch], dim=0)
    for data in batch:
        data.y = None
    batch_data = Batch.from_data_list(batch)
    batch_data.y = y
    return batch_data

# Load hyperparameters and the model
run_name = "fmo3d_secondary"
log_dir = f"runs/{run_name}"
model_path = f"{log_dir}/model.pt"
hparams_path = f"{log_dir}/hparams.pkl"

# Load hyperparameters (only for batch_size calculation)
if os.path.exists(hparams_path):
    with open(hparams_path, 'rb') as f:
        params = pickle.load(f)
else:
    # If there is no hparams.pkl, use the default parameter (replace it with the value during training)
    params = {
        "CONVS": 7,  # Sample value, needs to be replaced
        "LINEAR": 3,
        "DIM": 200,
        "ARCH": 0
    }

# Load the model (directly load the entire model object)
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()

# Load the test dataset
bsz = int((2**14) / params["DIM"])  # Consistent with the training
test_dataset = Dataset(is_train=False)
test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False, collate_fn=collate_fn)

# predict
all_preds = []
with torch.no_grad():
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        pred = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
        pred = torch.sigmoid(pred)  # Convert to probability [0, 1]
        all_preds.append(pred.cpu())
all_preds = torch.cat(all_preds, dim=0)  # shape [num_samples, 9]



# Load the original test CSV, including SMILES and true label
test_csv = pd.read_csv("single/openpom_test_dataset_secondary.csv")

# Read the label column name from label.txt
with open("single_secondary/labels.txt", "r", encoding="utf-8") as f:
    label_columns = [line.strip() for line in f if line.strip()]

# Keep nonStereoSMILES and true labels
true_df = test_csv[['nonStereoSMILES'] + label_columns].copy()
# Rename true labels to _true
true_df = true_df.rename(columns={col: f"{col}_true" for col in label_columns})

# Create a DataFrame for predicting probabilities and thresholding labels
pred_df = pd.DataFrame(
    all_preds.numpy(),
    columns=[f"{col}_probability" for col in label_columns]
)
thresh_df = (pred_df > 0.5).astype(int)
thresh_df.columns = [f"{col}_prediction" for col in label_columns]

# # Merge the original SMILES and the predicted results
# result_df = pd.concat([test_csv, pred_df, thresh_df], axis=1)

# Merge the original SMILES, true labels and the prediction results
result_df = pd.concat([true_df, pred_df, thresh_df], axis=1)
# save result
output_path = "odorpair_secondary_aroma_predictions.csv"
result_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
