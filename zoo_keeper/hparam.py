import ast
import re
import collections.abc

try:  # pragma: no cover
    from colorama import Fore

    BLUE, YELLOW, RESET = Fore.BLUE, Fore.YELLOW, Fore.RESET
except ImportError:  # pragma: no cover
    BLUE = YELLOW = RESET = ""

SPLIT_REGEX = re.compile(r",?(\w+)=")


def group(sequence):
    return zip(*[iter(sequence)] * 2)


class HParams(collections.abc.Mapping):
    """Class to hold a set of immutable hyperparameters as name-value pairs.

    You can override hyperparameter values by calling the `parse()` method, passing a
    string of comma separated `name=value` pairs. This is intended to make it possible
    to override any hyperparameter values from a single command-line flag to which
    the user passes 'hyper-param=value' pairs. It avoids having to define one flag for
    each hyperparameter.

    ```python
    class Hyperparameters(HParams):
        hidden_units = [32, 64, 128]
        learning_rate = 0.1

        @property
        def optimizer(self):
            return tf.keras.optimizers.Adam(self.learning_rate)

    hyper_parameters = Hyperparameters()

    hyper_parameters.parse("learning_rate=0.5,hidden_units=[32, 32, 32]")
    hyper_parameters.learning_rate  # -> 0.5
    hyper_parameters.hidden_units  # -> [32, 32, 32]
    hyper_parameters.optimizer  # -> <tf.keras.optimizers.Adam> with learning rate 0.5
    ```
    """

    def parse(self, value):
        """Override existing hyperparameter values, parsing new values from a string.

        # Arguments:
        values: String. Comma separated list of `name=value` pairs.

        # Returns:
        The `HParams` instance.

        # Raises:
        ValueError: If `values` cannot be parsed or a hyperparameter in `values`
            doesn't exist.
        """
        for key, value in group(filter(None, SPLIT_REGEX.split(value))):
            if not key in self.__iter__():
                raise ValueError(f"Unknown hyperparameter '{key}'")
            try:
                value = ast.literal_eval(value)
            except:
                raise ValueError(f"Could not parse '{value}'") from None
            object.__setattr__(self, key, value)

    def __iter__(self):
        return (k for k in self.__class__.__dict__.keys() if not k.startswith("_"))

    def __getitem__(self, item):
        if item in self.__iter__():
            return getattr(self, item)
        raise KeyError(item)

    def __len__(self):
        return len(list(self.__iter__()))

    def __setattr__(self, name, value):
        raise AttributeError("Hyperparameters are immutable, cannot assign to field.")

    def __str__(self):
        tab = "\n    "
        params = f",{tab}".join(
            [f"{BLUE}{k}{RESET}={YELLOW}{v}{RESET}" for k, v in sorted(self.items())]
        )
        return f"{self.__class__.__name__}({tab}{params}\n)"

    def __repr__(self):
        return self.__str__()
