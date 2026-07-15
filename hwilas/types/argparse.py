import logging
import argparse

logger = logging.getLogger(__name__)

class KeyValueAction(argparse.Action):
    """
    Argparse action to accept key:value pairs.
    These are stored in a dictionary.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, {})
        for value in values:
            key, val_str = value.split(':', 1)
            val = float(val_str)
            getattr(namespace, self.dest)[key] = val