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
    loss: Optional[Union[keras.losses.Loss, str]]
    optimizer: Union[keras.optimizers.Optimizer, str]
