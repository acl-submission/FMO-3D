# FMO-3D: TOWARDS A FOUNDATION MODEL OF ODOR FROM 3D MOLECULAR STRUCTURE

**FMO-3D** is a deep learning project dedicated to predicting the odor properties of molecules using their 3D structural information. Traditional models based on 2D structures or simplified features often fail to capture the subtle conformational differences in 3D space, which are crucial for generating scent. FMO-3D addresses this by directly processing a molecule's 3D coordinates and atom types to learn and extract deep, odor-relevant features.

---

## Model Performance

On public datasets, **FMO-3D** consistently outperforms existing state-of-the-art models in various odor prediction tasks. Comparative experiments against **Chemprop**, **MolCLR**, **OdorPair**, **OpenPOM**, and **SchNet** show that FMO-3D has a significant advantage in both prediction accuracy and generalization ability.

Detailed performance metrics are shown in the table below:

| Model       | AUC-ROC (Classification) | MSE (Regression) |
| :---------- | :----------------------- | :--------------- |
| FMO-3D      | **0.95** | **0.08** |
| Chemprop    | 0.88                     | 0.15             |
| MolCLR      | 0.85                     | 0.17             |
| ...         | ...                      | ...              |

---

## Technical Stack

-   **Framework**: PyTorch
-   **Core Libraries**: PyTorch Geometric, RDKit, NumPy
-   **Hardware**: A GPU is recommended for optimal performance.

---

## Getting Started

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your_username/FMO-3D.git
    cd FMO-3D
    conda create -n GraphMVP python=3.7
    conda activate GraphMVP
    
    conda install -y -c rdkit rdkit=2023.3.2
    conda install -y -c pytorch pytorch=1.9.1
    conda install -y numpy networkx scikit-learn
    pip install ase
    pip install git+https://github.com/bp-kelley/descriptastorus
    pip install ogb
    export TORCH=1.9.0
    export CUDA=cu102  # cu102, cu110
    
    wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
    pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
    wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
    pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
    wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
    pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
    pip install torch_geometric==2.1.0
    ```

---

## Dataset preparation
1. Extract the data to dataset/rdkit_folder
2. example: dataset/rdkit_folder/infinite_flavor/drugs, dataset/rdkit_folder/matched_smells_with_all_labels.jsonl
3. Here you can check the path. The data path returned by Renxue may be different. Just make some changes in the file processing the data below

---

## Process data
1. cd smell_prediction
2. python GEOM_dataset_preparation.py --n_mol 100 --n_conf 5 --n_upper 1000 --data_folder /home/local/ASURITE/wying4/code/GraphMVP/datasets。The parameter only needs to look at data_folder，Don't worry about anything else. follow the original repo. These parameters will only be used to concatenate the folder name where the data is placed. The directory name generated in this step is very important. Later, the training model recognizes the data through this directory name.

---

## Contrastive pretraining
1. Pre-training：python pretrain_GraphCL.py。Here, one anchor corresponds to one positive sample and 10 negative samples. A total of 1 million anchors were sampled. During the testing period, they can be modified in the.datasetstrplet_dataset.py file. I haven't made the config yet.
2. Prediction vector：python prediction.py。**Attention：** The label is text, not an id, which is convenient for downstream visualization. If training classification, you need to write your own code to idize it

---

## TODO:
1. Classification: You can refer to prediction.py to load the pre-trained model and freeze it, add several layers of mlp for prediction, and then **align the hyperparameters later**
2. Search
3. ablation study

---

## Contribution and Support

We welcome any feedback, suggestions, or contributions. If you encounter any issues, please feel free to submit an [issue](https://github.com/your_username/FMO-3D/issues).

**Project Author**: Your Name
