import argparse
import json
import os
import pickle
import random
from itertools import repeat
from os.path import join

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from datasets import allowable_features


def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data


def summarise():
    """ summarise the stats of molecules and conformers """
    dir_name = '{}/rdkit_folder'.format(data_folder)
    drugs_file = '{}/summary_drugs.json'.format(dir_name)

    with open(drugs_file, 'r') as f:
        drugs_summary = json.load(f)
    # expected: 304,466 molecules
    print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

    sum_list = []
    drugs_summary = list(drugs_summary.items())

    for smiles, sub_dic in tqdm(drugs_summary):
        ##### Path should match #####
        if sub_dic.get('pickle_path', '') == '':
            continue

        mol_path = join(dir_name, sub_dic['pickle_path'])
        with open(mol_path, 'rb') as f:
            mol_sum = {}
            mol_dic = pickle.load(f)
            conformer_list = mol_dic['conformers']
            conformer_dict = conformer_list[0]
            rdkit_mol = conformer_dict['rd_mol']
            data = mol_to_graph_data_obj_simple_3D(rdkit_mol)

            mol_sum['geom_id'] = conformer_dict['geom_id']
            mol_sum['num_edge'] = len(data.edge_attr)
            mol_sum['num_node'] = len(data.positions)
            mol_sum['num_conf'] = len(conformer_list)

            # conf['boltzmannweight'] a float for the conformer (a few rotamers)
            # conf['conformerweights'] a list of fine weights of each rotamer
            bw_ls = []
            for conf in conformer_list:
                bw_ls.append(conf['boltzmannweight'])
            mol_sum['boltzmann_weight'] = bw_ls
        sum_list.append(mol_sum)
    return sum_list


class Molecule3DDataset(InMemoryDataset):

    def __init__(self, root, n_mol, n_conf, n_upper, transform=None, seed=777,
                 pre_transform=None, pre_filter=None, empty=False, **kwargs):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)
        if 'smiles_copy_from_3D_file' in kwargs:  # for 2D Datasets (SMILES)
            self.smiles_copy_from_3D_file = kwargs['smiles_copy_from_3D_file']
        else:
            self.smiles_copy_from_3D_file = None

        self.root, self.seed = root, seed
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(Molecule3DDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('root: {},\ndata: {},\nn_mol: {},\nn_conf: {}'.format(
            self.root, self.data, self.n_mol, self.n_conf))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []
        data_smiles_list = []
        path_to_data = {}  # Store the mapping from pickle_path to the Data object directly

        dir_name = '{}/rdkit_folder'.format(data_folder)

        # Read matched_smells.json（anchors）
        drugs_file = '{}/matched_smells_with_single_label.jsonl'.format(dir_name)
        drugs_summary = {}
        with open(drugs_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                drugs_summary.update(item)

        # Get all the pickle paths
        all_pickle_paths = set()

        # Obtain the path from the anchor
        path_to_name = {}
        print('--------ying-------')
        print(len(drugs_summary.items()))
        for smiles, sub_dic in drugs_summary.items():
            if 'pickle_path' in sub_dic and sub_dic['pickle_path']:
                all_pickle_paths.add(sub_dic['pickle_path'])
                if 'Compound_Name' in sub_dic:
                    path_to_name[sub_dic['pickle_path']] = sub_dic['Compound_Name']
            if 'positive_samples' in sub_dic and sub_dic['positive_samples']:
                for pos_path in sub_dic['positive_samples']:
                    all_pickle_paths.add(pos_path[0])
            if 'negative_samples' in sub_dic and sub_dic['negative_samples']:
                for neg_path in sub_dic['negative_samples']:
                    all_pickle_paths.add(neg_path[0])

        print(f"A total of {len(all_pickle_paths)} unique pickle paths were found")
        # Process all molecules and construct mappings
        mol_idx, idx, notfound = 0, 0, 0

        # label for storing molecular formulas
        path_to_label = {}
        path_to_secondary = {}
        path_to_third = {}
        path_to_id = {} # Record the index of the data in data_list
        for pickle_path in tqdm(all_pickle_paths):
            mol_path = join(dir_name, pickle_path)
            try:
                with open(mol_path, 'rb') as f:

                    data = pickle.load(f)
                    rdkit_mol = data[0]['rd_mol']  # Obtain the rdkit mol object

                    # Obtain the first-level classification
                    if 'primary_aroma' in data[0] and data[0]['primary_aroma'] != []:
                        # Handle the first-level classification
                        path_to_label[pickle_path] = data[0]['primary_aroma']
                    else:
                        path_to_label[pickle_path] = 'unknown'  # If there is no first-level classification, set it to 'unknown'
                    # Obtain the secondary classification
                    if 'secondary_aroma' in data[0] and data[0]['secondary_aroma'] != []:
                        path_to_secondary[pickle_path] = data[0]['secondary_aroma']
                    else:
                        path_to_secondary[pickle_path] = 'unknown'  # If there is no secondary category, set it to 'unknown'

                    if 'smells_aroma' in data[0] and data[0]['smells_aroma'] != []:
                        path_to_third[pickle_path] = data[0]['smells_aroma']
                    else:
                        path_to_third[pickle_path] = 'unknown'  # If there is no third-level classification, set it to 'unknown'

                    # Convert to a Data object
                    data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    data.id = torch.tensor([idx])
                    data.mol_id = torch.tensor([mol_idx])

                    # Save the mapping from pickle_path to data directly
                    path_to_data[pickle_path] = data

                    # add to list
                    smiles = Chem.MolToSmiles(rdkit_mol)
                    data_smiles_list.append(smiles)
                    data_list.append(data)

                    path_to_id[pickle_path] = idx  # Record the index of the data in data_list

                    idx += 1
                    mol_idx += 1
            except Exception as e:
                print(f"An error occurred when processing the file {pickle_path}: {str(e)}")
                print(data)
                notfound += 1
                continue

        print(f"Successfully processed {len(data_list)} molecules")

        # Create a triplet relationship
        triplet_indices = []

        # Traverse all anchors
        for smiles, sub_dic in tqdm(drugs_summary.items()):
            if 'pickle_path' not in sub_dic or not sub_dic['pickle_path']:
                continue

            anchor_path = sub_dic['pickle_path']
            if anchor_path not in path_to_data:
                continue

            anchor_name = path_to_name[anchor_path]
            anchor_label = path_to_label[anchor_path]
            second_label = path_to_secondary[anchor_path]
            third_label = path_to_third[anchor_path]
            anchor_idx = path_to_id[anchor_path]  # Obtain the index of the data in the list

            # Obtain positive samples
            positive_paths = []
            if 'positive_samples' in sub_dic:
                positive_paths = sub_dic['positive_samples']
                positive_paths = positive_paths

            # Obtain negative samples
            negative_paths = []
            if 'negative_samples' in sub_dic:
                negative_paths = sub_dic['negative_samples']
                negative_paths = negative_paths

            # Make sure the path is in the mapping and obtain the index
            positive_indices = []
            for p in positive_paths:
                path = p[0]
                sim = p[1]
                if path in path_to_data:
                    pos_idx = path_to_id[path]
                    positive_indices.append((pos_idx, sim))

            negative_indices = []
            for n in negative_paths:
                path = n[0]
                sim = n[1]
                if path in path_to_data:
                    neg_idx = path_to_id[path]
                    negative_indices.append((neg_idx, sim))

            # Add a triplet relationship
            if positive_indices and negative_indices:
                triplet_indices.append((anchor_idx, positive_indices, negative_indices, anchor_name, anchor_label, second_label, third_label))

        print(f"Created the {len(triplet_indices)} triple relation")

        # Save the triplet relationship
        triplet_path = join(self.processed_dir, 'triplets.pt')
        torch.save(triplet_indices, triplet_path)

        # Process data and save it
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save the SMILES list
        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print(f'Save SMILES to {saver_path}')
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        # Save the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print(f"{notfound} molecules do not meet the requirements")
        print(f"{mol_idx} molecules have been treated")
        print(f"{idx} conformations have been processed")
        return


def load_SMILES_list(file_path):
    SMILES_list = []
    with open(file_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            SMILES_list.append(line.strip().decode())
    return SMILES_list


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum', type=bool, default=False, help='cal dataset stats')
    parser.add_argument('--n_mol', type=int, default=100, help='number of unique smiles/molecules')
    parser.add_argument('--n_conf', type=int, default=5, help='number of conformers of each molecule')
    parser.add_argument('--n_upper', type=int, default=1000, help='upper bound for number of conformers')
    parser.add_argument('--data_folder', default="../datasets", type=str)
    args = parser.parse_args()

    data_folder = args.data_folder

    if args.sum:
        sum_list = summarise()
        with open('{}/summarise.json'.format(data_folder), 'w') as fout:
            json.dump(sum_list, fout)

    else:
        n_mol, n_conf, n_upper = args.n_mol, args.n_conf, args.n_upper
        root_3d = '{}/GEOM_3D_nmol{}_nconf{}_nupper{}'.format(data_folder, n_mol, n_conf, n_upper)

        # Generate 3D Datasets (2D SMILES + 3D Conformer)
        Molecule3DDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper)

    ##### to data copy to SLURM_TMPDIR under the `datasets` folder #####
    '''
    wget https://dataverse.harvard.edu/api/access/datafile/4327252
    mv 4327252 rdkit_folder.tar.gz
    cp rdkit_folder.tar.gz $SLURM_TMPDIR
    cd $SLURM_TMPDIR
    tar -xvf rdkit_folder.tar.gz
    '''

    ##### for data pre-processing #####
    '''
    python GEOM_dataset_preparation.py --n_mol 100 --n_conf 5 --n_upper 1000 --data_folder datasets/fmo_3d
    '''
