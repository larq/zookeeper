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


def str_key_val(key, value, color=True):
    if callable(value):
        value = "<callable>"
    return f"{BLUE}{key}{RESET}={YELLOW}{value}{RESET}" if color else f"{key}={value}"


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

    _abc_methods = {"get", "items", "keys", "parse", "values"}

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
        hparams_keys = set(self.__iter__())
        for key, value in group(
            filter(
                None, SPLIT_REGEX.split(value.replace(",\n", ",").replace(", ", ","))
            )
        ):
            if not key in hparams_keys:
                raise ValueError(f"Unknown hyperparameter '{key}'")
            try:
                value = ast.literal_eval(value)
            except ValueError:
                # Parse as string if above raises ValueError. Note that
                # syntax errors will still raise an error.
                value = str(value)
            except:
                raise ValueError(f"Could not parse '{value}'") from None
            object.__setattr__(self, key, value)

    def _is_hparam(self, item):
        return item not in self._abc_methods and not item.startswith("_")

    def __iter__(self):
        return (item for item in self.__dir__() if self._is_hparam(item))

    def __getitem__(self, item):
        try:
            if self._is_hparam(item):
                return getattr(self, item)
        except:
            pass
        raise KeyError(item)

    def __len__(self):
        return len(list(self.__iter__()))

    def __setattr__(self, name, value):
        raise AttributeError("Hyperparameters are immutable, cannot assign to field.")

    def __str__(self):
        params = ",\n    ".join([str_key_val(k, v) for k, v in sorted(self.items())])
        return f"{self.__class__.__name__}(\n    {params}\n)"

    def __repr__(self):
        params = ",".join(
            [str_key_val(k, v, color=False) for k, v in sorted(self.items())]
        )
        return f"{self.__class__.__name__}({params})"
