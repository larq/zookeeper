from typing import Callable, Optional, Union, List

from tensorflow import keras
from zookeeper.dataset import Dataset
from zookeeper.task import Task
from zookeeper.model import Model
from zookeeper.preprocessing import Preprocessing


class Experiment(Task):
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
    loss: Union[keras.losses.Loss, str]
    optimizer: Union[keras.optimizers.Optimizer, str]
    learning_rate_schedule: Optional[Callable] = None
    callbacks: List[Union[keras.callbacks.Callback, Callable]] = []
