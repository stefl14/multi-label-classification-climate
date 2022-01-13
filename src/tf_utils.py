import keras
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf

K = keras.backend


def create_tokenized_ds(tokenizer,
        X, y, text_column: str = "description_text", model_name: str = "bert-base-cased", batch_size: int = 16
):
    """
    Create a tokenized batched tensorflow dataset from. This function is incomplete but the idea is to ensure
    a consistent pipeline between train and test.

    :param X:
    :param y:
    :param text_column:
    :param model_name:
    :param batch_size:
    :return:
    """
    y = np.array(y, dtype="int32")
    X = X[text_column]
    tokens = tokenizer(
        list(X),
        padding="max_length",
        truncation=True,
        return_tensors="tf",
        return_token_type_ids=False,
    )
    tokens_w_labels = tokens.copy()
    tokens_w_labels["target"] = y
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(tokens_w_labels), y))
    tf_dataset = tf_dataset.shuffle(len(tf_dataset)).batch(batch_size)
    return tf_dataset


class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(
            self.model.optimizer.learning_rate,
            self.model.optimizer.learning_rate * self.factor,
        )


def find_learning_rate(model, X, epochs: int = 1, min_rate: float = 10 ** -10, max_rate: float = 10 ** -1.5,
                       batch_size: int = 16):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X)) * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.learning_rate, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr])
    # reinitialise
    K.set_value(model.optimizer.learning_rate, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates: list, losses: list):
    """
    Plots loss over an epoch to decide on a learning rate after tuning
    any hyperparameters.
    :param rates:
    :param losses:
    :return:
    """
    plt.plot(rates, losses)
    plt.gca().set_xscale("log")
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


class OneCycleScheduler(keras.callbacks.Callback):
    """
    Implements one cycle learning rate schedule for speeding up training.
    """
    def __init__(
            self,
            iterations,
            max_rate,
            start_rate=None,
            last_iterations=None,
            last_rate=None,
    ):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return (rate2 - rate1) * (self.iteration - iter1) / (iter2 - iter1) + rate1

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(
                0, self.half_iteration, self.start_rate, self.max_rate
            )
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(
                self.half_iteration,
                2 * self.half_iteration,
                self.max_rate,
                self.start_rate,
            )
        else:
            rate = self._interpolate(
                2 * self.half_iteration,
                self.iterations,
                self.start_rate,
                self.last_rate,
            )
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, rate)
