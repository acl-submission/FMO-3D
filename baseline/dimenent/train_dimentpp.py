"""
1. The data format needs to be converted first. Use process_datasets.py to convert targets to labels of shape [n_samples, num_labels, 2].
2. Then train the model using train_dimentpp.py (modify training parameters and targets as needed).
"""

import os
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider
import random

# ==================== Logger setup ====================
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s (%(levelname)s): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

# ==================== Configuration ====================
# Modify the actual training parameters as needed
train_dataset_file = 'baseline_data/diment_datasets/fmo-3d/train_third_multilabel.npz'
test_dataset_file  = 'beseline_data/diment_datasets/fmo-3d/test_third_multilabel.npz'

# Automatically read target labels
labels_file = 'data/fmo-3d/third_labels.txt'
with open(labels_file, 'r', encoding='utf-8') as f:
    targets = [line.strip() for line in f if line.strip()]

level = 'third'
batch_size = 16
learning_rate = 1e-3
num_epochs = 200
save_interval = 1
evaluation_interval = 1

savedir = f'./fmo3d/{level}'
best_ckpt_file = os.path.join(savedir, 'best_model.ckpt')

# ==================== Set global randomness ====================
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ==================== Load data ====================
data_train = DataContainer(train_dataset_file, cutoff=5.0, target_keys=targets)
data_test  = DataContainer(test_dataset_file, cutoff=5.0, target_keys=targets)

train_provider = DataProvider(data_container=data_train, train_split=0.8, batch_size=batch_size, seed=seed)
# val_provider can be the same as train_provider
test_provider  = DataProvider(data_container=data_test, batch_size=1, seed=seed, is_test=True)

# ==================== Dataset generation ====================
def build_dataset(provider, split):
    idx_ds = provider.get_idx_dataset(split)
    ds = idx_ds.map(lambda idx: provider.idx_to_data_tf(idx, split), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = build_dataset(train_provider, 'train')
val_ds   = build_dataset(train_provider, 'val')
test_ds  = build_dataset(test_provider, 'test')

# ==================== Metrics & Model ====================
train_metrics = Metrics('train', targets)
val_metrics   = Metrics('val', targets)
test_metrics  = Metrics('test', targets)
model = DimeNetPP(num_labels=len(targets))
trainer = Trainer(model, learning_rate=learning_rate)

# ==================== Logging directory ====================
if not os.path.exists(savedir):
    os.makedirs(savedir)

metrics_best = {'mean_accuracy_val': 0.0, 'step': 0}
summary_writer = tf.summary.create_file_writer(savedir)

# ==================== Training loop ====================
steps_per_epoch = int(np.ceil(train_provider.nsamples['train'] / batch_size))
for epoch in range(1, num_epochs + 1):
    logger.info(f"Epoch {epoch}/{num_epochs} start")
    train_iter = iter(train_ds)
    for _ in range(steps_per_epoch):
        trainer.train_on_batch(train_iter, train_metrics)

    val_iter = iter(val_ds)
    for _ in range(int(np.ceil(train_provider.nsamples['val'] / batch_size))):
        trainer.test_on_batch(val_iter, val_metrics)

    logger.info(f"Epoch {epoch}: train_acc={train_metrics.mean_accuracy:.4f}, val_acc={val_metrics.mean_accuracy:.4f}")

    # Save the best model weights
    if val_metrics.mean_accuracy > metrics_best['mean_accuracy_val']:
        metrics_best['mean_accuracy_val'] = val_metrics.mean_accuracy
        metrics_best['step'] = epoch
        model.save_weights(best_ckpt_file)
        logger.info(f"New best model saved at epoch {epoch}")

    with summary_writer.as_default():
        tf.summary.scalar('train_mean_accuracy', train_metrics.mean_accuracy, step=epoch)
        tf.summary.scalar('val_mean_accuracy', val_metrics.mean_accuracy, step=epoch)

    train_metrics.reset_states()
    val_metrics.reset_states()

print("Training completed.")

# ==================== Prediction ====================
logger.info("Loading best model for prediction...")
model.load_weights(best_ckpt_file)

probs_list = []
pred_list = []
true_list = []
sample_ids = []

for i in range(test_provider.nsamples['test']):
    batch_inputs, batch_targets = test_provider.idx_to_data([i], split='test')  # batch_size=1
    probs = model(batch_inputs, training=False).numpy()  # shape [1, num_labels, 2]
    probs_label1 = probs[0, :, 1]  # extract probability for class 1
    pred_label = (probs_label1 > 0.5).astype(int)    # apply threshold
    true_label = batch_targets.numpy()[0, :, 1]       # true class 1 values

    probs_list.append(probs_label1)
    pred_list.append(pred_label)
    true_list.append(true_label)
    sample_ids.append(i)

# Convert to numpy arrays
probs_arr = np.stack(probs_list)  # [num_samples, num_labels]
pred_arr = np.stack(pred_list)
true_arr = np.stack(true_list)

# ==================== Save predictions to CSV ====================
data_dict = {'sample_id': sample_ids}
for i, t in enumerate(targets):
    data_dict[f'{t}_probability'] = probs_arr[:, i]
    data_dict[f'{t}_prediction'] = pred_arr[:, i]
    data_dict[f'{t}_true'] = true_arr[:, i]

df = pd.DataFrame(data_dict)
csv_file = os.path.join(savedir, 'test_results.csv')
df.to_csv(csv_file, index=False)
logger.info(f"Predictions, probabilities, and true labels saved to {csv_file}")

