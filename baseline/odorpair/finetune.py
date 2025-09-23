
import torch
from torch.utils.tensorboard import SummaryWriter
import torch_geometric as pyg
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torchmetrics
import tqdm
import copy
import uuid
import scipy.stats
from single.data import Dataset
import h5py
import numpy as np

torch.manual_seed(42)
auroc = torchmetrics.classification.MultilabelAUROC(num_labels=Dataset.num_classes())
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def save_embeddings_h5(model, loader, filename="embeddings.h5"):
    model.eval()
    all_embs, all_labels, all_smiles, all_file_paths = [], [], [], []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)

            # get embedding
            emb = model.gcn(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            all_embs.append(emb.cpu())
            all_labels.append(batch_data.y.cpu())

            # Collect smiles and file_path
            all_smiles.extend(batch_data.smiles)        # There is.smiles in Data
            all_file_paths.extend(batch_data.file_path) # There is.file_path in Data

    # Splicing
    all_embs = torch.cat(all_embs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_smiles = np.array(all_smiles, dtype="S")
    all_file_paths = np.array(all_file_paths, dtype="S")

    with h5py.File(filename, "w") as f:
        f.create_dataset("file_path", data=all_file_paths)
        f.create_dataset("smiles", data=all_smiles)
        f.create_dataset("embeddings", data=all_embs)
        f.create_dataset("labels", data=all_labels)



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

def do_train(params):
    print(params)
    model = SingleMoleculePredictor(
        num_convs=params["CONVS"],
        num_linear=params["LINEAR"],
        embedding_size=int(params["DIM"]),
        architecture=architectures[params["ARCH"]]
    )
    model = model.to(device)

    bsz = int((2**14) / (params["DIM"]))
    print(f"BSZ={bsz}")

    def collate_fn(batch):
        y = torch.stack([data.y for data in batch], dim=0)
        for data in batch:
            data.y = None
        batch_data = Batch.from_data_list(batch)
        batch_data.y = y
        return batch_data

    train_loader = DataLoader(
        Dataset(is_train=True), batch_size=bsz, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        Dataset(is_train=False), batch_size=bsz, shuffle=False, collate_fn=collate_fn
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["LR"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=params["DECAY"], total_iters=0.9 * params["STEPS"]
    )

    def do_train_epoch():
        model.train()
        losses = []
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            pred = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            y = batch_data.y.view(pred.shape).to(device)
            loss = loss_fn(pred, y)
            # print("pred shape:", pred.shape)
            # print("batch_data.y shape:", batch_data.y.shape)
            loss.backward()
            batch_size = batch_data.y.size(0)
            losses.append(loss * batch_size)
            optimizer.step()
        return torch.stack(losses).sum() / len(train_loader.dataset)

    def collate_test():
        model.eval()
        preds, ys = [], []
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            with torch.no_grad():
                pred = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            y = batch_data.y.view(pred.shape).to(device)
            preds.append(pred)
            ys.append(y)
        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)

    def get_test_loss():
        pred, y = collate_test()
        return loss_fn(pred, y)

    def get_auroc():
        pred, y = collate_test()
        return auroc(pred, y.int())

    run_name = 'openpom_secondary'
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')
    best = copy.deepcopy(model)
    for s in tqdm.tqdm(range(int(params["STEPS"]))):
        loss = do_train_epoch()
        scheduler.step()
        tl = get_test_loss()
        writer.add_scalars('Loss', {'train': loss, 'test': tl}, s)
        if tl < best_loss:
            best_loss = tl
            best = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1
            if patience > 20:
                break

    torch.save(best, f"{log_dir}/model.pt")
    # Generate h5 files (using test_loader or train_loader)
    save_embeddings_h5(best, test_loader, filename=f"{log_dir}/test_embeddings.h5")
    save_embeddings_h5(best, train_loader, filename=f"{log_dir}/train_embeddings.h5")

    metrics = {"auroc": get_auroc(), "completed": s}
    print(run_name, metrics, params, sep="\n")
    writer.add_hparams(params, metrics)
    writer.close()

architectures = ["GIN"]

def generate_params():
    distributions = {
        'STEPS': 200,  # Fixed value
        # 'LR': scipy.stats.loguniform(1e-5, 5e-4),
        'LR': 0.001,
        'DIM': 200,    # Fixed value
        "LINEAR": scipy.stats.randint(1, 6),
        "CONVS": scipy.stats.randint(3, 8),
        "DECAY": scipy.stats.loguniform(1e-4, .1),
        "ARCH": scipy.stats.randint(0, len(architectures))
    }

    params = {}
    for key, val in distributions.items():
        if hasattr(val, "rvs"):  # If it is a distributed object
            params[key] = val.rvs(1).item()
        else:  # If it is a fixed value
            params[key] = val
    return params


if __name__ == "__main__":
    do_train(generate_params())  # For single training sessions, cross-validation is disabled
