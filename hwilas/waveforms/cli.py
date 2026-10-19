"""Common waveform selection and typed generator arguments for command-line tools."""
import argparse
import json
from hwilas.types import KeyValueAction
from .plugins import normalise_approximant


def parse_approximant(value):
    try:
        return normalise_approximant(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def keyword_value(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def add_waveform_arguments(parser, *, dest='waveform_type', aliases=('--waveform-type',)):
    parser.add_argument('--approximant', *aliases, dest=dest, type=parse_approximant,
                        default='fastbhb', help='FastBHB, PhenomTHM or an installed external waveform provider')
    parser.add_argument('--orbits', '--orbit', dest='orbit', choices=['analytic','esa'], default='analytic')
    parser.add_argument('--f-ref', type=float, help='Phase reference frequency; defaults to the selected model convention')
    parser.add_argument('--data-time-origin', type=float, help='Absolute time of detector sample zero for ESA orbits')
    parser.add_argument('--generator-keywords', nargs='+', action=KeyValueAction,
                        type=keyword_value, default=None, metavar='KEY:VALUE')


def apply_waveform_config(args, cfg):
    from .plugins import prepare_generator, normalise_approximant
    settings = cfg['waveform']
    for attribute, key in [('waveform_type','approximant'),('orbit','orbit'),
                           ('f_ref','f_ref'),('data_time_origin','data_time_origin')]:
        value = getattr(args,attribute,None)
        if value is not None:
            settings[key] = value
    settings['approximant'] = normalise_approximant(settings.get('approximant','fastbhb'))
    keywords = {**settings.get('generator_keywords',{}), **(getattr(args,'generator_keywords',None) or {})}
    settings['generator_keywords'] = prepare_generator(settings['approximant'],
        orbit=settings.get('orbit','analytic'), data_time_origin=settings.get('data_time_origin'),
        start_time=0., end_time=4*86400., keywords=keywords)
    return cfg
