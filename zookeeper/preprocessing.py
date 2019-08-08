import abc


class Preprocessing(abc.ABC):
    """An abstract class to be used to define data preprocessing.

    We define a neural network to be a mapping between an input and an output, hence we
    define two abstract methods, an input (e.g the image for image classification), and
    an output (e.g the class label for image classification). We also define decoders,
    which allows use to customize the decoding, and kwargs, which allows us to pass
    information from the preprocessing to the model (e.g input size.)

    # Arguments
    features: A [`tfds.features.FeaturesDict`](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict)

    # Properties
    - `decoders`: Nested `dict` of [`Decoder`](https://www.tensorflow.org/datasets/api_docs/python/tfds/decode/Decoder)
        objects which allow to customize the decoding. The structure should match the
        feature structure, but only customized feature keys need to be present.
        See [the guide](https://www.tensorflow.org/datasets/decode) for more info.
    - `kwargs`: A `dict` that can be used to pass additional keyword arguments to a
        model function.
    """

    decoders = None
    kwargs = {}

    def __init__(self, features=None):
        self.features = features

    @abc.abstractmethod
    def inputs(self, data, training):
        """A method to define preprocessing for inputs.

        This method needs to be overwritten by all subclasses.

        # Arguments
        data: A dictionary of type {feature_name: Tensor}
        training: An optional `boolean` to define if preprocessing is called during training.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def outputs(self, data, training):
        """A method to define preprocessing for outputs.

        This method needs to be overwritten by all subclasses.

        # Arguments
        data: A dictionary of type {feature_name: Tensor}
        training: An optional `boolean` to define if preprocessing is called during training.
        """
        raise NotImplementedError("Must be implemented in subclasses.")
