import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import umap
from collections import Counter
from typing import Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# ---------- Tools ----------
def replace_unknown_to_UNK(label_list):
    if isinstance(label_list, str) and label_list.lower() == "unknown":
        return ["UNK"]
    if isinstance(label_list, list):
        new_labels = ["UNK" if str(i).lower() == "unknown" else str(i) for i in label_list]
        return list(set(new_labels))
    return [str(label_list)]


# ---------- 1. Data loading ----------
anchor_name = "anchor_name"
label_name = "label_name"
secondary_name = "second_class"

# train_data = pd.read_hdf('datasets/generated_embedding/train_embedding_with_label.h5', key='df')
# train_X = train_data.iloc[:, 0:300].values
# train_data[anchor_name] = train_data[anchor_name].apply(replace_unknown_to_UNK)
# train_y = train_data[anchor_name].apply(replace_unknown_to_UNK).values 
# train_y_first = train_data[label_name].values
# train_y_second = train_data[secondary_name].values

test_data = pd.read_hdf('datasets/generated_embedding/test_embedding_with_label.h5', key='df')
# test_X = test_data.iloc[:, 0:300].values

embedding_cols = [c for c in test_data.columns if c.startswith("emb_")]
test_X = test_data[embedding_cols].values.astype(float)

# Do normalization again
test_X_normalized = test_X / np.linalg.norm(test_X, axis=1, keepdims=True)
test_data[anchor_name] = test_data[anchor_name].apply(replace_unknown_to_UNK)
test_y = test_data[anchor_name].values
test_y_first = test_data[label_name].values
test_y_second = test_data[secondary_name].values

# # Normalized embedding vector
# train_X_normalized = train_X / np.linalg.norm(train_X, axis=1, keepdims=True)
# norms = np.linalg.norm(train_X_normalized, axis=1)
# print(f"Normalized vector norms: min={norms.min()}, max={norms.max()}, mean={norms.mean()}")

# Normalized embedding vector
test_X_normalized = test_X / np.linalg.norm(test_X, axis=1, keepdims=True)
norms = np.linalg.norm(test_X_normalized, axis=1)
# print(f"Normalized vector norms: min={norms.min()}, max={norms.max()}, mean={norms.mean()}")


# # ---------- 2. Load the compound_name similarity file ----------
# with open("/home/cc/Desktop/FMO-3D-main/compound_name_similarity.json", "r", encoding="utf-8") as f:
#     name_similarity = json.load(f)


def retrieval_with_label_sim(query_vec, query_name, query_label_list,
                             test_X, test_y, test_y_label_list,
                             ks=[3,5,7,10], threshold=0.0):
    """
    Only calculate the matching situation of the labels：
    - related: As long as one label matches, it is considered relevant
   
    """

    # Normalized query vector
    query_vec_normalized = query_vec / np.linalg.norm(query_vec)
    structure_scores = cosine_similarity([query_vec_normalized], test_X).flatten()

    results = []
   

    for i in range(len(test_X)):
        if test_y[i][0] == query_name:
            continue

        test_label = test_y_label_list[i]

        # ---------- Only calculate the primary label ----------
        matched_label = set(query_label_list) & set(test_label)
        matched_label_count = len(matched_label)

        related = matched_label_count > 0

        results.append({
            "index": i,
            "compound": test_y[i][0],
            "cos_sim": structure_scores[i],
            "related": related,
            "label_list": set(test_label)
        })

    # Sort by structural similarity
    results_sorted = sorted(results, key=lambda x: -x["cos_sim"])

    # return top-k
    top_k_dict = {k: results_sorted[:k] for k in ks}

    return top_k_dict


# ---------- 4. Evaluate the top-k retrieval of the query ----------
def compute_metrics(top_k_results, query_label_list):
    metrics = {}
    query_label_set = set(query_label_list)

    for k, items in top_k_results.items():
        retrieved_relevant = 0
        top_k_label_set = set()  # Each k is calculated separately
       

        for idx, r in enumerate(items, start=1):
            if r["related"]:
                retrieved_relevant += 1
            
            # Accumulate the union of top-k labels
            top_k_label_set.update(r["label_list"])

        precision = retrieved_relevant / k if k > 0 else 0.0
        recall = len(top_k_label_set & query_label_set) / len(query_label_set) if len(query_label_set) > 0 else 0.0
        matched_label_count = len(top_k_label_set & query_label_set)

        metrics[k] = {
            "precision": precision,
            "recall": recall,
            "matched_label_count": matched_label_count,
            "retrieved_relevant_compounds": retrieved_relevant,
            "total_query_labels": len(query_label_set)
        }

    return metrics


# ---------- 5. Evaluate each compound in the test ----------
ks = [3, 5, 7, 10]
all_metrics = {k: {"precision": [], "recall": [],  "matched_label_count": [], "retrieved_relevant_compounds": [], "total_query_labels": []} for k in ks}

for random_idx in range(len(test_X)):
    query_vec = test_X[random_idx]
    query_name = test_y[random_idx][0]
    query_label_first = test_y_first[random_idx]
    query_label_second = test_y_second[random_idx]


    top_k_results = retrieval_with_label_sim(
        query_vec, query_name, query_label_second,
        test_X_normalized, test_y, test_y_second,
        ks=ks
    )

    metrics = compute_metrics(top_k_results, query_label_second)

    # Save the metrics for each query
    for k, m in metrics.items():
        all_metrics[k]["precision"].append(m["precision"])
        all_metrics[k]["recall"].append(m["recall"])
       

# Calculate the average index
for k in ks:
    avg_precision = np.mean(all_metrics[k]["precision"])
    avg_recall = np.mean(all_metrics[k]["recall"])
  
    print(f"Top-{k} Average: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}")





