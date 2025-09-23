import torch
from torch.utils.data import Dataset
from ase.db import connect
from ase.neighborlist import NeighborList
import schnetpack.properties as properties
import schnetpack.properties as structure

def atoms_to_schnet_input(atoms, cutoff=5.0):
    """Convert ASE Atoms into SchNet input dictionaries"""
    Z = torch.LongTensor(atoms.numbers)

    # Establish a neighbor table
    cutoffs = [cutoff / 2.0] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    idx_i = []
    idx_j = []
    rij_list = []

    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            idx_i.append(i)
            idx_j.append(j)
            rij_list.append(atoms.positions[j] + offset @ atoms.get_cell() - atoms.positions[i])

    idx_i = torch.LongTensor(idx_i)
    idx_j = torch.LongTensor(idx_j)
    Rij = torch.FloatTensor(rij_list)

    return {
        properties.Z: Z,
        properties.Rij: Rij,
        properties.idx_i: idx_i,
        properties.idx_j: idx_j
    }

class AseDbDataset(Dataset):
    def __init__(self, db_path, cutoff=5.0):
        self.db_path = db_path
        self.cutoff = cutoff

        # Save the index first
        with connect(db_path) as conn:
            self.ids = [row.id for row in conn.select()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        with connect(self.db_path) as conn:
            row = conn.get(self.ids[idx])
            atoms = row.toatoms()

        inputs = atoms_to_schnet_input(atoms, cutoff=self.cutoff)

        # If you have tags, such as row.data["target"]
        target = None
        if "target" in row.data:
            target = torch.tensor(row.data["target"], dtype=torch.float32)

        return inputs, target

import torch

def schnet_collate_fn(batch):
    """
    Collate function for batching molecules for SchNet (without labels).

    Args:
        batch: list of samples, each is a dict/tuple (inputs)
            - inputs: dict with keys:
                structure.Z, structure.Rij, structure.idx_i, structure.idx_j

    Returns:
        dict with:
            - structure.Z: Tensor[N_total]
            - structure.Rij: Tensor[N_total_pairs, 3]
            - structure.idx_i: Tensor[N_total_pairs]
            - structure.idx_j: Tensor[N_total_pairs]
            - "batch": Tensor[N_total]  # Which molecule does each atom belong to
    """
    z_list, rij_list, idx_i_list, idx_j_list, batch_index_list, y_list = [], [], [], [], [], []
    n_atoms_cum = 0  # Used for offset idx_i/idx_j

    for mol_idx, (inputs, y) in enumerate(batch):
        z = inputs[structure.Z]          # [N_atoms]
        r_ij = inputs[structure.Rij]     # [N_pairs, 3]
        idx_i = inputs[structure.idx_i]  # [N_pairs]
        idx_j = inputs[structure.idx_j]  # [N_pairs]
        y_list.append(y)

        # Atomic characteristics
        z_list.append(z)

        # Atomic pair information, pay attention to idx offset
        idx_i_list.append(idx_i + n_atoms_cum)
        idx_j_list.append(idx_j + n_atoms_cum)
        rij_list.append(r_ij)

        # batch index, which molecule each atom belongs to
        batch_index_list.append(torch.full((z.size(0),), mol_idx, dtype=torch.long))

        # Update offset
        n_atoms_cum += z.size(0)

    return {
        structure.Z: torch.cat(z_list, dim=0),
        structure.Rij: torch.cat(rij_list, dim=0),
        structure.idx_i: torch.cat(idx_i_list, dim=0),
        structure.idx_j: torch.cat(idx_j_list, dim=0),
        "batch": torch.cat(batch_index_list, dim=0),
        "y": torch.stack(y_list, dim=0)
    }