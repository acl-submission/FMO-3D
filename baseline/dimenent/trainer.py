import tensorflow as tf
import tensorflow_addons as tfa
from .schedules import LinearWarmupExponentialDecay


class Trainer:
    def __init__(self, model, learning_rate=1e-3, warmup_steps=None,
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm

       
        if warmup_steps is not None:
            self.learning_rate = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate
            )

       
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)

       
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)  # [batch, num_labels, 2]
            loss = self.loss_fn(targets, preds)
        grads = tape.gradient(loss, self.model.trainable_weights)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        metrics.update_state(loss, preds, targets)
        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)
        loss = self.loss_fn(targets, preds)
        metrics.update_state(loss, preds, targets)
        return loss

    @tf.function
    def predict_on_batch(self, dataset_iter):
        inputs, _ = next(dataset_iter)
        preds = self.model(inputs, training=False)
        return preds
