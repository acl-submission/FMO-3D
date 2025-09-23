import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import args
from models import GNN, AutoEncoder, SchNet, VariationalAutoEncoder
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from torch_geometric.data import Batch
import pandas as pd

from datasets import Molecule3DMaskingDataset, PredictionDataset
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args.device = 0
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    # Let's use this name for now
    args.dataset = 'GEOM_3D_nmol100_nconf5_nupper1000'
    data_root = '../datasets/{}/'.format(args.dataset)

    base_dataset = Molecule3DMaskingDataset(
        root=data_root,
        dataset=args.dataset,
        mask_ratio=args.SSL_masking_ratio
    )

    prediction_dataset = PredictionDataset(data_root, base_dataset)
    loader = DataLoader(
        prediction_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=prediction_dataset.collate_triplets
    )

    # Considering whether to add a projection layer after the model for Contrastive learning
    molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
    optimizer = optim.AdamW(molecule_model_3D.parameters(), lr=args.lr, weight_decay=args.decay)

    model_dict = torch.load('saved_model/best_model.pth', map_location=device)
    molecule_model_3D.load_state_dict(model_dict)

    all_results = []
    molecule_model_3D.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(loader):
            anchors, labels, label_names, anchor_names, filter_infos, second_classes, thrird_classes = batch_data
            anchors = [data.to(device) for data in anchors]
            anchors_batch = Batch.from_data_list(anchors)
            anchor_tensor = molecule_model_3D(anchors_batch.x[:, 0], anchors_batch.positions, batch=anchors_batch.batch)

            anchor_np = anchor_tensor.detach().cpu().numpy()
            df = pd.DataFrame(anchor_np)
            df['label'] = labels
            df['label_name'] = label_names
            df['second_class'] = second_classes
            df['third_class'] = thrird_classes
            df['anchor_name'] = anchor_names
            df['filter_info'] = filter_infos

            all_results.append(df)

        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_hdf('embedding_with_label.h5', key='df', mode='w')