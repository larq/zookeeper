from typing import Callable, List, Optional, Union

from tensorflow import keras

from zookeeper.tf.dataset import Dataset
from zookeeper.tf.model import Model
from zookeeper.tf.preprocessing import Preprocessing


class Experiment:
    """
    A wrapper around a Keras experiment. Subclasses must implement their
    training loop in `run`.
    """

    # Nested components
    dataset: Dataset
    preprocessing: Preprocessing
    model: Model

    # Parameters
    epochs: int
    batch_size: int
    metrics: List[Union[keras.metrics.Metric, Callable, str]] = []
    loss: Optional[Union[keras.losses.Loss, str]]
    optimizer: Union[keras.optimizers.Optimizer, str]
    learning_rate_schedule: Optional[Callable] = None
    callbacks: List[Union[keras.callbacks.Callback, Callable]] = []
