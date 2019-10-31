from functools import partial
from inspect import getfullargspec

from zookeeper import Component


def pass_training_kwarg(function, training=False):
    if "training" in getfullargspec(function).args:
        return partial(function, training=training)
    return function


class Preprocessing(Component):
    """
    An wrapper around `tf.data` preprocessing.
    """

    def inputs(self, data, training):
        """
        A method to define preprocessing for inputs.
        This method or `__call__` needs to be overwritten by all subclasses.

        Arguments:
            data:
                A dictionary of type {feature_name: Tensor}.
            training:
                An optional `bool` to indicate whether the data is training
                data.
        Returns:
            A tensor of processed inputs.
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    def outputs(self, data, training):
        """
        A method to define preprocessing for outputs.
        This method or `__call__` needs to be overwritten by all subclasses.

        Arguments:
            data:
                A dictionary of type {feature_name: Tensor}.
            training:
                An optional `bool` to indicate whether the data is training
                data.
        Returns:
            A tensor of processed outputs.
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(self, data, training=False):
        """
        Apply Preprocessing.

        Arguments:
            data:
                A dictionary of type {feature_name: Tensor}.
            training:
                An optional `bool` to indicate whether the data is training
                data.
        Returns:
            A pair of processed inputs and outputs.
        """

        input_fn = pass_training_kwarg(self.inputs, training=training)
        output_fn = pass_training_kwarg(self.outputs, training=training)
        return input_fn(data), output_fn(data)
