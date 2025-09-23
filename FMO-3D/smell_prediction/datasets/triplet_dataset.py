import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TripletDataset(Dataset):
    """
    PyTorch DataLoader, a dataset class adaptation standard specifically designed for handling triplet data
    """
    def __init__(self, root, main_dataset):
        
        self.root = root
        self.main_dataset = main_dataset
        self.samples = []
        self.triplet_indices = torch.load(self.root + '/processed/triplets.pt')

        positive_pairs = set()
        sampled_data = []
        for idx, triplet in enumerate(self.triplet_indices):
        
            anchor_idx, positive_indices, negative_indices, _, _, _, _ = triplet
            
            # Positive sample
            pos_indices = []
            # Avoid not reaching the threshold and not selecting a positive sample pair
            for (pos_idx, sim) in positive_indices:
                if (anchor_idx, pos_idx) not in positive_pairs:
                    pos_indices.append((pos_idx, sim))
                    positive_pairs.add((anchor_idx, pos_idx))
                    positive_pairs.add((pos_idx, anchor_idx))

            for pos_idx in pos_indices:
                # Negative sample sampling
                n_pos = 1
                n_neg = min(len(negative_indices), n_pos * 10)
                if len(negative_indices) > n_neg:
                    neg_indices = random.sample(negative_indices, n_neg)
                else:
                    neg_indices = negative_indices
                sampled_data.append((anchor_idx, pos_idx, neg_indices))
        logging.info("triplets loaded: %d", len(sampled_data))
        self.samples = random.sample(sampled_data, min(len(sampled_data), 1000000))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_indices = self.samples[idx]
        anchor_data = self.main_dataset.get(anchor_idx)
        pos_data = self.main_dataset.get(pos_idx[0])
        neg_data = [self.main_dataset.get(neg_idx[0]) for neg_idx in neg_indices]
        return anchor_data, pos_data, neg_data

    def collate_triplets(self, samples):
        anchors, positives, negatives = map(list, zip(*samples))
    
        return anchors, positives, negatives
