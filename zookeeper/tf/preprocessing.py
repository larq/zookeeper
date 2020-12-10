import functools
import inspect
from typing import Dict, Optional, Tuple, Union

import tensorflow as tf
import tensorflow_datasets as tfds

from zookeeper.core.field import Field  # type: ignore


def pass_training_kwarg(function, training=False):
    if "training" in inspect.signature(function).parameters:
        return functools.partial(function, training=training)
    return function


class Preprocessing:
    """A wrapper around `tf.data` preprocessing."""

    decoders: Optional[Dict[str, Union[tfds.decode.Decoder, Dict]]] = Field(None)

    # The shape of the processed input. Must match the output of `input()`.
    input_shape: Tuple[int, int, int] = Field()

    def input(self, data, training) -> tf.Tensor:
        """A method to define preprocessing for model input. This method or `__call__`
        needs to be overwritten by all subclasses.

        Arguments:
            data:
                A dictionary of type {feature_name: tf.Tensor}.
            training:
                An optional `bool` to indicate whether the data is training
                data.
        Returns:
            A tensor of processed input, with shape `self.input_shape`.
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    def output(self, data, training) -> tf.Tensor:
        """A method to define preprocessing for model output. This method or `__call__`
        needs to be overwritten by all subclasses.

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
        """Apply Preprocessing.

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
