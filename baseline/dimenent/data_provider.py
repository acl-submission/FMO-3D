from collections import OrderedDict
import numpy as np
import tensorflow as tf
from .data_container import index_keys

class DataProvider:
    def __init__(self, data_container, train_split=0.8, batch_size=1, seed=None, randomized=False, is_test=False):
        """
        Initialize the DataProvider, which can handle train/validation splits or be used for test-only predictions.

        Args:
            data_container: DataContainer instance (training or test set)
            train_split: Fraction of data used for training (only effective for training)
            batch_size: Batch size
            seed: Random seed
            randomized: Whether to shuffle the training data
            is_test: Whether used only for prediction (no train/validation split)
        """
        self.data_container = data_container
        self.batch_size = batch_size
        self._random_state = np.random.RandomState(seed=seed)
        self.is_test = is_test

        n_total = len(data_container)
        if not is_test:
            ntrain = int(n_total * train_split)
            nvalid = n_total - ntrain
            self.nsamples = {'train': ntrain, 'val': nvalid, 'test': n_total}
            idx_all = np.arange(n_total)
            if randomized:
                idx_all = self._random_state.permutation(idx_all)
            self.idx = {
                'train': idx_all[:ntrain],
                'val': idx_all[ntrain:],
                'test': idx_all
            }
        else:
            self.nsamples = {'test': n_total}
            self.idx = {'test': np.arange(n_total)}

        self.idx_in_epoch = {k: 0 for k in self.nsamples.keys()}

        # Input data types
        from .data_container import index_keys
        self.dtypes_input = OrderedDict()
        self.dtypes_input['Z'] = tf.int32
        self.dtypes_input['R'] = tf.float32
        for key in index_keys:
            self.dtypes_input[key] = tf.int32
        self.dtype_target = tf.float32

        # Input data shapes
        self.shapes_input = {k: [None] if k != 'R' else [None, 3] for k in self.dtypes_input.keys()}
        self.shape_target = [None, len(data_container.target_keys), 2]

    def get_batch_idx(self, split):
        """Return indices for the next batch from the specified split."""
        start = self.idx_in_epoch[split]
        if self.idx_in_epoch[split] >= self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]
        return self.idx[split][start:end]

    def idx_to_data(self, idx, split, return_flattened=False):
        """Convert a list of indices to actual data from the DataContainer."""
        batch = self.data_container[idx]
        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            inputs_targets.append(tf.constant(batch['targets'], dtype=tf.float32))
            return inputs_targets
        else:
            inputs = {key: tf.constant(batch[key], dtype=dtype) for key, dtype in self.dtypes_input.items()}
            targets = tf.constant(batch['targets'], dtype=tf.float32)
            return (inputs, targets)

    def idx_to_data_tf(self, idx, split):
        """TensorFlow wrapper to convert indices to tensors, supporting tf.py_function."""
        dtypes_flattened = list(self.dtypes_input.values()) + [self.dtype_target]
        inputs_targets = tf.py_function(
            lambda idx: self.idx_to_data(idx.numpy(), split, return_flattened=True),
            inp=[idx], Tout=dtypes_flattened)
        inputs = {k: inputs_targets[i] for i, k in enumerate(self.dtypes_input.keys())}
        for k in self.dtypes_input.keys():
            inputs[k].set_shape(self.shapes_input[k])
        targets = inputs_targets[-1]
        targets.set_shape(self.shape_target)
        return (inputs, targets)

    def get_idx_dataset(self, split='test'):
        """Create a tf.data.Dataset that yields batches of indices for the specified split."""
        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)
        return tf.data.Dataset.from_generator(generator, output_types=tf.int32, output_shapes=[None])

