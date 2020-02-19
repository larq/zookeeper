from typing import Optional, Union

from tensorflow import keras

from zookeeper.core.field import ComponentField, Field
from zookeeper.tf.dataset import Dataset
from zookeeper.tf.preprocessing import Preprocessing


class Experiment:
    """
    A wrapper around a Keras experiment. Subclasses must implement their
    training loop in `run`.
    """

    # Nested components
    dataset: Dataset = ComponentField()
    preprocessing: Preprocessing = ComponentField()
    model: keras.models.Model = ComponentField()

    # Parameters
    epochs: int = Field()
    batch_size: int = Field()
    loss: Optional[Union[keras.losses.Loss, str]] = Field()
    optimizer: Union[keras.optimizers.Optimizer, str] = Field()
