import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import AseDbDataset, schnet_collate_fn
from schnet_classify import SchNet_Classify
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import schnetpack.properties as structure
from sklearn.metrics import precision_recall_fscore_support

from dataloader import AseDbDataset, schnet_collate_fn
from schnet_baseline.schnet_classify import SchNet_Classify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_num = 2
# ============ Data loading ============
train_db = "./baseline_data/train_dataset_primary.db"
test_db = "./baseline_data/test_dataset_primary.db"

train_dataset = AseDbDataset(train_db, cutoff=5.0)
test_dataset = AseDbDataset(test_db, cutoff=5.0)

train_loader = DataLoader(
    train_dataset, batch_size=batch_num, shuffle=True, collate_fn=schnet_collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_num, shuffle=False, collate_fn=schnet_collate_fn
)

# ============ model ============
model = SchNet_Classify(num_labels=12).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============ Training cycle ============
best_train_loss = float("inf")
num_epochs = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        # Take inputs and batch directly from batch dict
        batch_inputs = {
            structure.Z: batch[structure.Z].to(device),
            structure.Rij: batch[structure.Rij].to(device),
            structure.idx_i: batch[structure.idx_i].to(device),
            structure.idx_j: batch[structure.idx_j].to(device),
            "batch": batch["batch"].to(device)
        }
        batch_idx = batch["batch"].to(device)
        y = batch["y"].float().to(device)

        optimizer.zero_grad()
        # Native SchNet forward receives dict
        out = model(batch_inputs)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)

    train_loss /= len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")

    # Save the best model
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), "best_schnet_model.pt")
        print(f"Saved best model at epoch {epoch+1}, train_loss {train_loss:.4f}")

# ============ Testing phase ============
print("Loading best model for evaluation...")
model.load_state_dict(torch.load("best_schnet_model.pt"))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        # Take inputs and batch directly from batch dict
        batch_inputs = {
            structure.Z: batch[structure.Z].to(device),
            structure.Rij: batch[structure.Rij].to(device),
            structure.idx_i: batch[structure.idx_i].to(device),
            structure.idx_j: batch[structure.idx_j].to(device),
            "batch": batch["batch"].to(device)
        }
        batch_idx = batch["batch"].to(device)
        y = batch["y"].float().to(device)

        optimizer.zero_grad()
        # Native SchNet forward receives dict
        out = model(batch_inputs, batch_num)
        probs = torch.sigmoid(out)

        preds = (probs > 0.5).int().cpu()
        all_preds.append(preds)
        all_labels.append(y.cpu().int())

all_preds = torch.cat(all_preds, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="samples", zero_division=0
)

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1:        {f1:.4f}")
