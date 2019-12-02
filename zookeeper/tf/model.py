from abc import ABC, abstractmethod
from typing import Tuple

from tensorflow import keras


class Model(ABC):
    """
    A wrapper around a Keras model. Subclasses must implement `build` to
    build and return a Keras model.
    """

    @abstractmethod
    def build(self, input_shape: Tuple[int, int, int]) -> keras.models.Model:
        raise NotImplementedError
