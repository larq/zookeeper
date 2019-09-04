from zookeeper import Preprocessing
from unittest.mock import Mock


class PreprocessingSimple(Preprocessing):
    def inputs(self, data):
        pass

    def outputs(self, data):
        pass


class PreprocessingWithTraining(Preprocessing):
    _assert_training = False

    def inputs(self, data, training):
        assert training == self._assert_training

    def outputs(self, data, training):
        assert training == self._assert_training


def test_preprocessing_without_training_arg():
    prepro = PreprocessingSimple()
    prepro.inputs = Mock()
    prepro.outputs = Mock()

    prepro("data")
    prepro.inputs.assert_called_with("data")
    prepro.outputs.assert_called_with("data")


def test_preprocessing_with_training_arg():
    prepro = PreprocessingWithTraining()

    assert prepro._assert_training == False
    prepro("data", training=False)

    prepro._assert_training = True
    prepro("data", training=True)
