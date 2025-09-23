import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def replace_unknown_to_UNK(label_list):
    # If it is a string and unknown, return ['UNK'] directly.
    if isinstance(label_list, str) and label_list.lower() == "unknown":
        return ["UNK"]
    # If it is a list, process each element
    if isinstance(label_list, list):
        new_labels = []
        for i in label_list:
            if str(i).lower() == "unknown":
                new_labels.append("UNK")
            else:
                new_labels.append(str(i))
        # duplicate removal
        return list(set(new_labels))
    # In other cases, directly convert to a string
    return [str(label_list)]

# Select several levels of classification
label_name = "label_name"
# label_name = "second_class"
# label_name = "third_class"

# train
train_data = pd.read_hdf('train_embedding_with_label.h5', key='df')
train_X = train_data.iloc[:, 0:300].values
train_data[label_name] = train_data[label_name].apply(replace_unknown_to_UNK)
train_y = train_data[label_name].values

# test
test_data = pd.read_hdf('test_embedding_with_label.h5', key='df')
test_X = test_data.iloc[:, 0:300].values
test_data[label_name] = test_data[label_name].apply(replace_unknown_to_UNK)
test_y = test_data[label_name].values


# Merge all tags fit
all_y = np.concatenate([train_y, test_y])
mlb = MultiLabelBinarizer()
mlb.fit(all_y)

# transform respectively
train_y_encoded = mlb.transform(train_y)
test_y_encoded = mlb.transform(test_y)

# Convert to tensor
train_X = torch.tensor(train_X, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
train_y_encoded = torch.tensor(train_y_encoded, dtype=torch.float32)
test_y_encoded = torch.tensor(test_y_encoded, dtype=torch.float32)


# 5. Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

input_dim = 300
hidden_dim = 128
num_classes = len(mlb.classes_)
model = MLP(input_dim, hidden_dim, num_classes)

# 6. Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

torch.manual_seed(42)

# 7. train
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_X)
    loss = criterion(outputs, train_y_encoded)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 8. test
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(test_X, dtype=torch.float32))
    probs = logits.sigmoid().cpu().numpy()
    preds = (logits.sigmoid() > 0.5).cpu().numpy()  # Multi-label binary classification
    # test_y_encoded It is already one-hot
    precision = precision_score(test_y_encoded, preds, average='macro', zero_division=0)
    recall = recall_score(test_y_encoded, preds, average='macro', zero_division=0)
    f1 = f1_score(test_y_encoded, preds, average='macro', zero_division=0)

    # Calculation per-class AUC
    per_class_aucs = []
    skipped = []
    for j in range(test_y_encoded.shape[1]):
        if len(np.unique(test_y_encoded[:, j])) < 2:
            skipped.append(j)
            continue
        auc = roc_auc_score(test_y_encoded[:, j], probs[:, j])
        per_class_aucs.append(auc)
    macro_auc = float(np.mean(per_class_aucs)) if per_class_aucs else float('nan')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"auc-roc: {macro_auc:.4f}")