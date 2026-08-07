import argparse
import logging

logger = logging.getLogger(__name__)


class KeyValueAction(argparse.Action):
    """
    Argparse action to accept key:value pairs.

    Values are stored in a dictionary. If this action is configured with
    ``type=...`` in ``add_argument``, each value is converted with that
    callable before being stored.
    """

    def __init__(self, option_strings, dest, **kwargs):
        # Preserve caller-provided converter for values, while forcing argparse
        # to keep incoming tokens as strings (e.g. "target_fs:0.2").
        self.value_type = kwargs.pop('type', None)
        super().__init__(option_strings, dest, type=str, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, {})

        if isinstance(values, str):
            values = [values]

        converter = self.value_type if self.value_type is not None else str
        for value in values:
            if ':' not in value:
                raise argparse.ArgumentError(
                    self,
                    f"Invalid key:value pair '{value}'. Expected format key:value.",
                )

            key, val_str = value.split(':', 1)
            try:
                converted_value = converter(val_str)
            except (TypeError, ValueError) as exc:
                converter_name = getattr(converter, '__name__', str(converter))
                raise argparse.ArgumentError(
                    self,
                    f"Failed to convert value '{val_str}' for key '{key}' using {converter_name}.",
                ) from exc

            getattr(namespace, self.dest)[key] = converted_value