import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from collections import Counter

class PredictionDataset(Dataset):
    """
    PyTorch DataLoader, a dataset class adaptation standard specifically designed for handling triplet data
    """
    def __init__(self, root, main_dataset):
        
        self.root = root
        self.main_dataset = main_dataset
        self.samples = []
        self.triplet_indices = torch.load(self.root + '/processed/triplets.pt')

        label_dic = {}
        label_id = 0
        total_label_dic = {}
        total_label_id = 0
        
        for idx, triplet in enumerate(self.triplet_indices):
        
            anchor_idx, pos_indices, _, anchor_name, anchor_label, second_class, third_class = triplet
            
            counter = Counter(anchor_label)

            # Obtain the element with the highest occurrence frequency and its frequency
            most_common = counter.most_common(1)[0]
            most_common_value, frequency = most_common
            
            if most_common_value not in label_dic:
                label_dic[most_common_value] = label_id
                label_id += 1

            # Filter based on similarity
            if anchor_idx not in total_label_dic:
                total_label_dic[anchor_idx] = total_label_id
                total_label_id += 1
            for (pos_idx, sim) in pos_indices:
                if sim >= 0.8:
                    total_label_dic[pos_idx] = total_label_dic[anchor_idx]

            self.samples.append((anchor_idx, label_dic[most_common_value], anchor_label, anchor_name, total_label_dic[anchor_idx], second_class, third_class))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_idx, label, label_name, anchor_name, filter_info, second_class, third_class = self.samples[idx]
        anchor_data = self.main_dataset.get(anchor_idx)
        return anchor_data, label, label_name, anchor_name, filter_info, second_class, third_class

    def collate_triplets(self, samples):
        anchors, labels, label_names, anchor_names, filter_infos, second_classes, third_classes = map(list, zip(*samples))
        return anchors, labels, label_names, anchor_names, filter_infos, second_classes, third_classes
