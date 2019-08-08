import abc


class Preprocessing(abc.ABC):
    """ An abstract Preprocessing class that contains the abstract contains the
    methods required for Preprocessing.

    We define a neural network to be a mapping between an input and an output.
    hence we define to abstract methods, an input (e.g the image for image
    classification), and an output (e.g the class label for image classification).
    We also define decoders, which allows use to customize the decoding, and kwargs,
    which allows one to pass information from the preprocessing to the model (e.g
    input size.)
    """

    decoders = None
    kwargs = {}

    def __init__(self, features=None):
        self.features = features

    @abc.abstractmethod
    def inputs(self, data, training):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def outputs(self, data, training):
        raise NotImplementedError("Must be implemented in subclasses.")
