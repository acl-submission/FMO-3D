import torch
import torch.nn as nn
from torch_scatter import scatter
from schnetpack.representation.schnet import SchNet
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff


class SchNet_Classify(nn.Module):
    def __init__(self, num_labels=11, hidden_dim=10, cutoff=5.0):
        super().__init__()

        # SchNet backbone
        self.model = SchNet(
            n_atom_basis=hidden_dim,
            n_interactions=3,
            radial_basis=GaussianRBF(n_rbf=30, cutoff=cutoff),
            cutoff_fn=CosineCutoff(cutoff=cutoff)
        )

        self.readout = "mean" 

        self.num_labels = num_labels
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) 
            for _ in range(num_labels)
        ])

    def forward(self, batch_inputs):
        """
        Args:
            z: [N_atoms] Atomic number
            pos: [N_atoms, 3] Atomic coordinates
            batch: [N_atoms] Which molecule does each atom belong to

        Returns:
            logits: [N_mols, num_labels] Multi-label classification logits
        """
        # --- SchNet backbone ---
        out = self.model(batch_inputs)
        h = out["scalar_representation"]  # [N_atoms, hidden_dim]

        # --- Polymerize to the molecular level ---
        batch_idx = batch_inputs["batch"]  # [N_atoms]
        h = scatter(h, batch_idx, dim=0, reduce="mean")

        # --- Multi-category headers ---
        logits = [clf(h) for clf in self.classifiers]  # list of [N_mols, 1]
        logits = torch.cat(logits, dim=1)              # [N_mols, num_labels]

        return logits
