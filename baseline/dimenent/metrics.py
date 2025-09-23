import numpy as np
import tensorflow as tf


class Metrics:
    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets  
        self.ex = ex

       
        self.loss_metric = tf.keras.metrics.Mean(name=f'loss_{tag}')
       
        self.accuracy_metrics = [tf.keras.metrics.CategoricalAccuracy(name=f'accuracy_{t}') for t in targets]
        self.precision_metrics = [tf.keras.metrics.Precision(name=f'precision_{t}') for t in targets]
        self.recall_metrics = [tf.keras.metrics.Recall(name=f'recall_{t}') for t in targets]

    def update_state(self, loss, predictions, targets, nsamples=None):
      
        self.loss_metric.update_state(loss)
        for i, (acc, prec, rec) in enumerate(zip(self.accuracy_metrics, self.precision_metrics, self.recall_metrics)):
            acc.update_state(targets[:, i, :], predictions[:, i, :])
            prec.update_state(targets[:, i, :], predictions[:, i, :])
            rec.update_state(targets[:, i, :], predictions[:, i, :])

    def reset_states(self):
       
        self.loss_metric.reset_states()
        for metric_list in [self.accuracy_metrics, self.precision_metrics, self.recall_metrics]:
            for m in metric_list:
                m.reset_states()

    @property
    def mean_accuracy(self):
        return np.mean([m.result().numpy() for m in self.accuracy_metrics])

    @property
    def mean_precision(self):
        return np.mean([m.result().numpy() for m in self.precision_metrics])

    @property
    def mean_recall(self):
        return np.mean([m.result().numpy() for m in self.recall_metrics])

    def result(self):
       
        res = {
            f'loss_{self.tag}': self.loss_metric.result().numpy(),
            f'mean_accuracy_{self.tag}': self.mean_accuracy,
            f'mean_precision_{self.tag}': self.mean_precision,
            f'mean_recall_{self.tag}': self.mean_recall,
        }
      
        for i, t in enumerate(self.targets):
            res[f'accuracy_{t}_{self.tag}'] = self.accuracy_metrics[i].result().numpy()
            res[f'precision_{t}_{self.tag}'] = self.precision_metrics[i].result().numpy()
            res[f'recall_{t}_{self.tag}'] = self.recall_metrics[i].result().numpy()
        return res
