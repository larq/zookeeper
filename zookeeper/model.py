from abc import ABC, abstractmethod

from tensorflow import keras
from zookeeper import Component


class Model(Component, ABC):
    """
    A wrapper around a Keras model. Subclasses must implement `build` to
    build and return a Keras model.
    """

    @abstractmethod
    def build(self) -> keras.models.Model:
        pass
