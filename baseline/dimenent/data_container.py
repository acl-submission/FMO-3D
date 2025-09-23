import numpy as np
import scipy.sparse as sp

index_keys = ["batch_seg", "idnb_i", "idnb_j", "id_expand_kj",
              "id_reduce_ji", "id3dnb_i", "id3dnb_j", "id3dnb_k"]


class DataContainer:
    def __init__(self, filename, cutoff, target_keys):
        """
        Initialize the DataContainer and load molecular structures and multi-label targets.

        Args:
            filename: Path to the .npz file containing molecule data and labels
            cutoff: Distance cutoff for neighboring atoms
            target_keys: List of target keys, e.g., ['label1', 'label2', ..., 'label48']
        """
        data_dict = np.load(filename, allow_pickle=True)
        self.cutoff = cutoff
        self.target_keys = target_keys

        # Load molecular structure data
        for key in ['id', 'N', 'Z', 'R']:
            if key in data_dict:
                setattr(self, key, data_dict[key])
            else:
                setattr(self, key, None)

        # Load label data, expected shape [num_samples, 48, 2] as one-hot encoding
        if 'labels' in data_dict:
            self.targets = data_dict['labels']  # Directly load preprocessed labels
        else:
            # If labels are scattered across target_keys, stack them to [num_samples, len(target_keys), 2]
            self.targets = np.stack([data_dict[key] for key in self.target_keys], axis=1)

        if self.N is None:
            self.N = np.zeros(len(self.targets), dtype=np.int32)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

        assert self.R is not None, "Molecular coordinates R must be provided"

    def _bmat_fast(self, mats):
        """Fast construction of block-diagonal sparse matrices."""
        new_data = np.concatenate([mat.data for mat in mats])
        ind_offset = np.zeros(1 + len(mats))
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))])
        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1:] + indptr_offset[i] for i in range(len(mats))])
        return sp.csr_matrix((new_data, new_indices, new_indptr))

    def __len__(self):
        """Return the number of samples."""
        return self.targets.shape[0]

    def __getitem__(self, idx):
        """Retrieve sample data for the given indices."""
        if isinstance(idx, (int, np.int64)):
            idx = [idx]

        data = {}
        data['targets'] = self.targets[idx]  # Shape [batch, 48, 2]
        data['id'] = self.id[idx] if self.id is not None else None
        data['N'] = self.N[idx]
        data['batch_seg'] = np.repeat(np.arange(len(idx), dtype=np.int32), data['N'])
        adj_matrices = []

        data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
        data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)

        nend = 0
        for k, i in enumerate(idx):
            n = data['N'][k]  # Number of atoms in the molecule
            nstart = nend
            nend = nstart + n

            if self.Z is not None:
                data['Z'][nstart:nend] = self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]]

            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            data['R'][nstart:nend] = R

            Dij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_matrices.append(sp.csr_matrix(Dij <= self.cutoff))
            adj_matrices[-1] -= sp.eye(n, dtype=bool)

        # Construct adjacency matrix
        adj_matrix = self._bmat_fast(adj_matrices)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape)
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()

        # Target and source nodes for edges
        data['idnb_i'] = edgeid_to_target
        data['idnb_j'] = edgeid_to_source

        # Triplet indices k->j->i
        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
        id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
        id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
        id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]

        # Filter out i->j->i triplets
        id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
        data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
        data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
        data['id3dnb_k'] = id3ynb_k[id3_y_to_d]

        # Interaction edge indices
        data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
        return data

