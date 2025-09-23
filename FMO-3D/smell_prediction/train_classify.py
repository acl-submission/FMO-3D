import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from sklearn.metrics import precision_recall_fscore_support
from schnet_baseline.schnet import SchNet
from config import args
from torch.utils.data import DataLoader
from datasets import Molecule3DMaskingDataset, PredictionDataset


epochs = 200
task = "label_name"   # It can also be changed to "second_class" or "third_class"

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

    molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
    optimizer = optim.AdamW(molecule_model_3D.parameters(), lr=args.lr, weight_decay=args.decay)

    criterion = nn.BCEWithLogitsLoss()

    # ================== train Part ==================
    molecule_model_3D.train()
    for epoch in range(epochs):
        for step, batch_data in enumerate(loader):
            anchors, labels, label_names, anchor_names, filter_infos, second_classes, third_classes = batch_data
            anchors = [data.to(device) for data in anchors]
            anchors_batch = Batch.from_data_list(anchors)
            pred = molecule_model_3D(anchors_batch.x[:, 0], anchors_batch.positions, batch=anchors_batch.batch)

            # Select different tags according to the task
            if task == "label_name":
                target = label_names.to(device).float()
            elif task == "second_class":
                target = second_classes.to(device).float()
            elif task == "third_class":
                target = third_classes.to(device).float()
            else:
                raise ValueError(f"Unknown task {task}")

            loss = criterion(pred.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 5 == 0:
                print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")

    # save model
    torch.save(molecule_model_3D.state_dict(), f"schnet_{task}.pt")

    # ================== test part ==================
    results = []
    molecule_model_3D.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for step, batch_data in enumerate(loader):
            anchors, labels, label_names, anchor_names, filter_infos, second_classes, third_classes = batch_data
            anchors = [data.to(device) for data in anchors]
            anchors_batch = Batch.from_data_list(anchors)
            pred = molecule_model_3D(anchors_batch.x[:, 0], anchors_batch.positions, batch=anchors_batch.batch)

            if task == "label_name":
                target = label_names.to(device).float()
            elif task == "second_class":
                target = second_classes.to(device).float()
            elif task == "third_class":
                target = third_classes.to(device).float()

            preds = (torch.sigmoid(pred.squeeze()) > 0.5).long().cpu().numpy()
            labels_np = target.cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # Calculation precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"  # If there are multiple classes, it can be changed to "macro"
    )

    print(f"Test Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
