import pandas as pd
import numpy as np
import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants 
from openpom.models.mpnn_pom import MPNNPOMModel 
from datetime import datetime
from openpom.utils.data_utils import get_class_imbalance_ratio
from openpom.models.mpnn_pom import MPNNPOMModel
from sklearn.metrics import accuracy_score, f1_score
from rdkit import Chem
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
# ignore PyTorch's FutureWarning warnings('ignore') category=FutureWarning) Set environment variables to suppress PyTorch warnings os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
# 1. Tag file
with open('datasets/fmo3d_datasets/aroma_vocabularies.json', 'r') as f:
    vocab_data = json.load(f)

# levels = ['primary', 'secondary', 'third']  # 三个级别
levels = ['third']
data_root = 'datasets/fmo3d_datasets'
results_root = 'fmo3d_results'
os.makedirs(results_root, exist_ok=True)

# SMILES check function
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Train and predict each level in a loop
for level in levels:
    print(f"\n=== Processing {level} aroma ===")
    if level == 'third':
        TASKS = vocab_data[f'smells_aroma']['vocabulary']
    else:
        TASKS = vocab_data[f'{level}_aroma']['vocabulary']

   
    # Data path
    train_data_path = os.path.join(data_root, f'openpom_train_dataset_{level}.csv')
    test_data_path = os.path.join(data_root, f'openpom_test_dataset_{level}.csv')
    model_dir = f'model_checkpoints_for_{level}'
    os.makedirs(model_dir, exist_ok=True)

    # Read the training data
    train_df = pd.read_csv(train_data_path)
    train_df = train_df[train_df['nonStereoSMILES'].apply(is_valid_smiles)]
    print(f"Valid SMILES in training data: {len(train_df)}")

    # Check the task list
    missing_tasks = [task for task in TASKS if task not in train_df.columns]
    if missing_tasks:
        raise ValueError(f"Tasks {missing_tasks} not found in training dataset columns")

    # Characterization
    featurizer = GraphFeaturizer()
    X_train = featurizer.featurize(train_df['nonStereoSMILES'].values)
    y_train = train_df[TASKS].values
    train_dataset = dc.data.NumpyDataset(X_train, y_train)

    # Divide the training validation set
    from sklearn.model_selection import train_test_split
    train_indices, valid_indices = train_test_split(range(len(train_dataset)), test_size=0.1, random_state=42)
    train_dataset = dc.data.NumpyDataset(X_train[train_indices], y_train[train_indices])
    valid_dataset = dc.data.NumpyDataset(X_train[valid_indices], y_train[valid_indices])
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(valid_dataset)}")

    # Category imbalance
    train_ratios = get_class_imbalance_ratio(train_dataset)

    # Model initialization
    n_tasks = len(TASKS)
    model = MPNNPOMModel(
        n_tasks=n_tasks,
        batch_size=128,
        learning_rate=0.001,
        class_imbalance_ratio=train_ratios,
        loss_aggr_type='sum',
        node_out_feats=100,
        edge_hidden_feats=75,
        edge_out_feats=100,
        num_step_message_passing=5,
        mpnn_residual=True,
        message_aggregator_type='sum',
        mode='classification',
        number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM,
        n_classes=1,
        readout_type='set2set',
        num_step_set2set=3,
        num_layer_set2set=2,
        ffn_hidden_list=[392, 392],
        ffn_embeddings=256,
        ffn_activation='relu',
        ffn_dropout_p=0.12,
        ffn_dropout_at_input_no_act=False,
        weight_decay=1e-5,
        self_loop=False,
        optimizer_name='adam',
        log_frequency=32,
        model_dir=model_dir,
        device_name='cpu'
    )

    # train
    nb_epoch = 200
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    start_time = datetime.now()
    for epoch in range(1, nb_epoch+1):
        loss = model.fit(train_dataset, nb_epoch=1, max_checkpoints_to_keep=1, deterministic=False, restore=epoch>1)
        train_scores = model.evaluate(train_dataset, [metric])['roc_auc_score']
        valid_scores = model.evaluate(valid_dataset, [metric])['roc_auc_score']
        print(f"epoch {epoch}/{nb_epoch} ; loss = {loss}; train_scores = {train_scores}; valid_scores = {valid_scores}")
    model.save_checkpoint()
    print(f"✅ Model saved to {model_dir}")

    # test
    test_df = pd.read_csv(test_data_path)
    test_df = test_df[test_df['nonStereoSMILES'].apply(is_valid_smiles)]
    print(f"Valid SMILES in test data: {len(test_df)}")

    missing_test_tasks = [task for task in TASKS if task not in test_df.columns]
    if missing_test_tasks:
        raise ValueError(f"Tasks {missing_test_tasks} not found in test dataset columns")

    X_test = featurizer.featurize(test_df['nonStereoSMILES'].values)
    y_test = test_df[TASKS].values if TASKS[0] in test_df.columns else None
    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    model.restore()
    predictions = model.predict(test_dataset)
    y_pred = (predictions > 0.5).astype(int)

    # save CSV
    output_df = pd.DataFrame({'nonStereoSMILES': test_df['nonStereoSMILES'].values})
    for i, task in enumerate(TASKS):
        if y_test is not None:
            output_df[f'{task}_true'] = y_test[:, i]
        output_df[f'{task}_probability'] = predictions[:, i]
        output_df[f'{task}_prediction'] = y_pred[:, i]

    output_path = os.path.join(results_root, f'{level}_aroma_predictions.csv')
    output_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
