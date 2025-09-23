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

from datasets import TripletDataset,  MoleculeGraphCLMaskingDataset
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(args, molecule_model_3D, device, loader, optimizer):
    print("starting training")
    start_time = time.time()

    if molecule_model_3D is not None:
        molecule_model_3D.train()

    if args.verbose:
        batch_iter = tqdm(loader)
    else:
        batch_iter = loader

    best_loss = float('inf')
    total_loss = 0.0
    num_batches = 0

    for step, batch_data in tqdm(enumerate(batch_iter)):
        # Use dataloader to obtain a batch of triplet data
        anchors, positives, negatives = batch_data

        number_neg = len(negatives[0])  # The number of negative samples corresponding to each anchor
        # Move the data to the device
        anchors = [data.to(device) for data in anchors]
        positives = [data.to(device) for data in positives]
        # negatives: [batch_size, number_neg]，Each element is Data
        negatives = [[data.to(device) for data in neg_list] for neg_list in negatives]

        # # Merge into large images for batch processing
        anchors_batch = Batch.from_data_list(anchors)
        positives_batch = Batch.from_data_list(positives)

        # negatives: [B, number_neg]，Flatten first and then combine the batches
        negatives_flat = [item for sublist in negatives for item in sublist]  # [B*number_neg]

        negatives_batch = Batch.from_data_list(negatives_flat)

        optimizer.zero_grad()

        # 1. Calculate the embedding of anchorpositive
        anchor_tensor = molecule_model_3D(anchors_batch.x, anchors_batch.edge_index, anchors_batch.edge_attr)  # [B*N_atom, D] -> [B, D] Through readout
        anchor_tensor = global_mean_pool(anchor_tensor, anchors_batch.batch)   # [B, D]
        # print(len(anchor_tensor), len(anchor_tensor[0]))
        positive_tensor = molecule_model_3D(positives_batch.x, positives_batch.edge_index, positives_batch.edge_attr)
        positive_tensor = global_mean_pool(positive_tensor, positives_batch.batch)  # [B, D]
        # print(len(positive_tensor), len(positive_tensor[0]))

        # 3. negatives
        negative_tensor_flat = molecule_model_3D(negatives_batch.x, negatives_batch.edge_index, negatives_batch.edge_attr)
        negative_tensor_flat = global_mean_pool(negative_tensor_flat, negatives_batch.batch)  # [B*number_neg, D]

        negative_tensor = negative_tensor_flat.view(len(anchors), number_neg, -1)  # [B, number_neg, D]

        # Calculate the contrastive loss
        # --- INFO NCE ---
        pos_sim = F.cosine_similarity(anchor_tensor, positive_tensor, dim=-1)
        anchor_expand = anchor_tensor.unsqueeze(1)  # [B, 1, D]
        neg_sim = F.cosine_similarity(anchor_expand, negative_tensor, dim=-1)

        pos_sim = torch.exp(pos_sim / args.T)
        neg_sim = torch.exp(neg_sim / args.T)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=-1)))

        # ===== NT-Xent Loss (SimCLR) =====
        # z = torch.cat([anchor_tensor, positive_tensor], dim=0)  # [2B, D]
        # z = F.normalize(z, dim=-1)  # Unit

        # # Similarity matrix [2B, 2B]
        # sim = torch.matmul(z, z.T) / args.T

        # # Remove the diagonals (oneself and oneself)
        # mask = torch.eye(sim.size(0), device=sim.device).bool()
        # sim.masked_fill_(mask, -1e9)

        # # Construct the label: index of the positive sample in the batch
        # # For example, the positive sample of anchor[i] is positive[i]
        # B = anchor_tensor.size(0)
        # labels = torch.arange(B, device=sim.device)
        # labels = torch.cat([labels + B, labels], dim=0)  # [2B]

        # # Cross-entropy loss
        # loss = F.cross_entropy(sim, labels)

        # --- NCE Loss ---
        # anchor_tensor:   [B, D]
        # positive_tensor: [B, D]
        # negative_tensor: [B, N, D]

        # L2 normalization (optional, enhancing numerical stability)
        # anchor_norm = F.normalize(anchor_tensor, dim=-1)
        # positive_norm = F.normalize(positive_tensor, dim=-1)
        # negative_norm = F.normalize(negative_tensor, dim=-1)

        # # Calculate the similarity of positive samples
        # pos_sim = torch.sum(anchor_norm * positive_norm, dim=-1)  # [B]

        # # Calculate the negative sample similarity
        # anchor_expand = anchor_norm.unsqueeze(1)                 # [B, 1, D]
        # neg_sim = torch.sum(anchor_expand * negative_norm, dim=-1)  # [B, N]

        # # sigmoid as a binary classification probability
        # pos_loss = -torch.log(torch.sigmoid(pos_sim))            # [B]
        # neg_loss = -torch.sum(torch.log(1 - torch.sigmoid(neg_sim)), dim=-1)  # [B]

        # # NCE all loss
        # loss = (pos_loss + neg_loss).mean()


        loss = loss.mean()  # Average loss
        total_loss += loss.item()
        num_batches += 1

        loss.backward()
        optimizer.step()

        if step % 200 == 1:
            logging.info('Training loss: {:.4f}'.format(loss))

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    if avg_loss < best_loss:
        best_loss = avg_loss
        # Here, you can save the model or record the optimal loss, etc
        torch.save(molecule_model_3D.state_dict(), 'ablation_model/openpom_GraphCL_GNN_best_model.pth')
    logging.info('Training loss: {:.4f},  Training time: {:.2f}s'.format(avg_loss, time.time() - start_time))

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    # Let's use this name for now
    args.dataset = 'train_GEOM_3D_nmol100_nconf5_nupper1000'
    data_root = '../datasets_openpom/{}/'.format(args.dataset)

    # base_dataset = MoleculeGraphCLMaskingDataset(
    #     root=data_root,
    #     dataset=args.dataset,
    #     mask_ratio=args.SSL_masking_ratio
    # )

    base_dataset = MoleculeGraphCLMaskingDataset(
        root=data_root,
        dataset=args.dataset,
        mask_ratio=args.SSL_masking_ratio
    )
    
    trip_dataset = TripletDataset(data_root, base_dataset)
    loader = DataLoader(
        trip_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trip_dataset.collate_triplets
    )
    
    print("GNN SchNet")
    molecule_model_3D = GNN(
        num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio)

    print("GNN initialized")
    molecule_model_3D.to(device)
    print("to device")
    optimizer = optim.AdamW(molecule_model_3D.parameters(), lr=args.lr, weight_decay=args.decay)
    print("optim initialized")
    # Evaluate the initial loss before training
    # molecule_model_3D.eval()
    # total_loss = 0.0
    # num_batches = 0
    # with torch.no_grad():
    #     for batch_data in loader:
    #         anchors, positives, negatives = batch_data
    #         number_neg = len(negatives[0])
    #         anchors = [data.to(device) for data in anchors]
    #         positives = [data.to(device) for data in positives]
    #         negatives = [[data.to(device) for data in neg_list] for neg_list in negatives]
    #         anchors_batch = Batch.from_data_list(anchors)
    #         positives_batch = Batch.from_data_list(positives)
    #         negatives_flat = [item for sublist in negatives for item in sublist]
    #         negatives_batch = Batch.from_data_list(negatives_flat)
    #         anchor_tensor = molecule_model_3D(anchors_batch.x[:, 0], anchors_batch.positions, batch=anchors_batch.batch)
    #         positive_tensor = molecule_model_3D(positives_batch.x[:, 0], positives_batch.positions, batch=positives_batch.batch)
    #         negative_tensor_flat = molecule_model_3D(negatives_batch.x[:, 0], negatives_batch.positions, batch=negatives_batch.batch)
    #         negative_tensor = negative_tensor_flat.view(len(anchors), number_neg, -1)
    #         pos_sim = F.cosine_similarity(anchor_tensor, positive_tensor, dim=-1)
    #         anchor_expand = anchor_tensor.unsqueeze(1)
    #         neg_sim = F.cosine_similarity(anchor_expand, negative_tensor, dim=-1)
    #         pos_sim = torch.exp(pos_sim / args.T)
    #         neg_sim = torch.exp(neg_sim / args.T)
    #         loss = -torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=-1)))
    #         loss = loss.mean()
    #         total_loss += loss.item()
    #         num_batches += 1
    # init_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    # logging.info('Initial loss before training: {:.4f}'.format(init_loss))

    for epoch in range(1, args.epochs + 1):
        logging.info('---epoch {}-----'.format(epoch))
        print("---epoch {}-----".format(epoch))
        train(args, molecule_model_3D, device, loader, optimizer)
