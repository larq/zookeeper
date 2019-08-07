import abc


class Preprocessing(abc.ABC):
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
