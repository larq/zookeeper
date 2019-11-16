from functools import partial
from inspect import signature
from typing import Tuple

import tensorflow as tf

from zookeeper.component import Component


def pass_training_kwarg(function, training=False):
    if "training" in signature(function).parameters:
        return partial(function, training=training)
    return function


class Preprocessing(Component):
    """A wrapper around `tf.data` preprocessing."""

    def input(self, data, training) -> tf.Tensor:
        """
        A method to define preprocessing for model input. This method or
        `__call__` needs to be overwritten by all subclasses.

        Arguments:
            data:
                A dictionary of type {feature_name: tf.Tensor}.
            training:
                An optional `bool` to indicate whether the data is training
                data.
        Returns:
            A tensor of processed input.
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    def output(self, data, training) -> tf.Tensor:
        """
        A method to define preprocessing for model output. This method or
        `__call__` needs to be overwritten by all subclasses.

        Arguments:
            data:
                A dictionary of type {feature_name: tf.Tensor}.
            training:
                An optional `bool` to indicate whether the data is training
                data.
        Returns:
            A tensor of processed output.
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(self, data, training=False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply Preprocessing.

        Arguments:
            data:
                A dictionary of type {feature_name: tf.Tensor}.
            training:
                An optional `bool` to indicate whether the data is training
                data.
        Returns:
            A pair of processed input and output.
        """

        input_fn = pass_training_kwarg(self.input, training=training)
        output_fn = pass_training_kwarg(self.output, training=training)
        return input_fn(data), output_fn(data)
