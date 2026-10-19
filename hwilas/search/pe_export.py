"""Parameter-only handoff to BBHx/PhenomHM; no data or PSD preparation."""
import hashlib
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

logger = logging.getLogger(__name__)
REQUIRED = ('mass1', 'mass2', 'spin1z', 'spin2z', 'distance', 'tc',
            'eclipticlongitude', 'eclipticlatitude', 'inclination',
            'polarization', 'coa_phase')


def plain(value):
    """Preserve HDF scalars and arrays as portable YAML values."""
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(Path(path).resolve()), 'sha256': digest.hexdigest()}


def read_event(path, row_index):
    """Select a zero-based row in the same descending-statistic order as the CSV."""
    with h5py.File(path, 'r') as source:
        if 'optimised_stat' not in source:
            raise ValueError('Missing optimised_stat for CSV row ordering')
        statistics = source['optimised_stat'][:]
        if statistics.ndim != 1:
            raise ValueError('Statistics must be a one-dimensional column')
        if row_index < 0 or row_index >= len(statistics):
            raise ValueError(f'Row index {row_index} outside catalogue with {len(statistics)} rows')
        selected = int(np.argsort(-statistics, kind='stable')[row_index])
        columns = {k: v for k, v in source.items() if isinstance(v, h5py.Dataset)}
        if any(v.shape != statistics.shape for v in columns.values()):
            raise ValueError('Event columns must have identical one-dimensional shapes')
        row = {k: plain(v[selected]) for k, v in columns.items()}
        if set(REQUIRED) - row.keys():
            raise ValueError(f'Missing physical columns: {sorted(set(REQUIRED) - row.keys())}')
        root = {k: plain(v) for k, v in source.attrs.items()}
        metadata = dict(root)
        if 'round_metadata' in source:
            if 'search_round' not in row or str(int(row['search_round'])) not in source['round_metadata']:
                raise ValueError('Missing selected event round provenance')
            metadata.update({k: plain(v) for k, v in
                             source['round_metadata'][str(int(row['search_round']))].attrs.items()})
    return row, metadata, root


def export_pe_seed(parameters, row_index, out, target_time_origin,
                   native_phase_convention=None):
    row, metadata, root = read_event(parameters, row_index)
    for key in REQUIRED:
        if not math.isfinite(float(row[key])):
            raise ValueError(f'{key} must be finite')
    for key in ('data_time_origin', 'f_ref', 'waveform_model', 'orbit'):
        if key not in metadata:
            raise ValueError(f'Missing source metadata: {key}')
    if not all(math.isfinite(float(v)) for v in
               (metadata['data_time_origin'], metadata['f_ref'], target_time_origin)):
        raise ValueError('Time origins and reference frequency must be finite')
    if not row['mass1'] >= row['mass2'] > 0 or row['distance'] <= 0:
        raise ValueError('Require mass1 >= mass2 > 0 and positive distance (Mpc)')
    if any(abs(row[k]) > 1 for k in ('spin1z', 'spin2z')):
        raise ValueError('Aligned spins must be in [-1, 1]')
    if not -math.pi/2 <= row['eclipticlatitude'] <= math.pi/2 or not 0 <= row['inclination'] <= math.pi:
        raise ValueError('Invalid latitude or inclination in radians')
    model = metadata['waveform_model'].lower()
    thm = model in ('phenomthm', 'imrphenomthm', 'phenomthm+tdispline')
    if not thm and model != 'bbhx':
        raise ValueError(f'Unsupported waveform convention: {model}')
    if float(metadata['f_ref']) < 0 or (thm and float(metadata['f_ref']) == 0):
        raise ValueError('Invalid source reference frequency')
    if not thm and native_phase_convention is None:
        raise ValueError('External BBHx providers require an explicit native phase convention')
    phase = (math.pi/2-row['coa_phase'] if thm or native_phase_convention == 'pi-over-2-minus-phase'
             else row['coa_phase'])
    seed = dict(m1=row['mass1'], m2=row['mass2'], chi1z=row['spin1z'], chi2z=row['spin2z'],
                distance=row['distance'], inc=row['inclination'],
                lam=row['eclipticlongitude'] % (2*math.pi), beta=row['eclipticlatitude'],
                psi=row['polarization'] % math.pi, phi_ref=phase % (2*math.pi),
                t_ref=row['tc']+(float(metadata['data_time_origin'])-target_time_origin),
                f_ref=0. if thm else float(metadata['f_ref']), injection_frame='SSB')
    provenance = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'row_index': int(row_index), 'row_order': 'descending optimised_stat, stable ties (CSV order)', 'source': fingerprint(parameters),
        'native_search': row, 'native_metadata': metadata, 'catalogue_metadata': root,
        'native_units': {'mass1': 'detector/SSB solar masses', 'mass2': 'detector/SSB solar masses',
                         'distance': 'Mpc', 'tc': 'detector seconds from native data_time_origin',
                         'spin1z': 'dimensionless', 'spin2z': 'dimensionless',
                         'eclipticlongitude': 'SSB ecliptic radians', 'eclipticlatitude': 'SSB ecliptic radians',
                         'inclination': 'radians', 'polarization': 'radians', 'coa_phase': 'radians',
                         'f_ref': 'Hz', 'f_lower': 'Hz', 'data_time_origin': 'absolute orbit seconds'},
        'exported_seed': dict(seed),
        'derived_seed': {'M': seed['m1']+seed['m2'], 'q': seed['m2']/seed['m1'],
                         'q_convention': 'm2/m1 <= 1; Hwilas bank q uses m1/m2 >= 1'},
        'units': {'m1': 'solar masses, detector/SSB frame', 'm2': 'solar masses, detector/SSB frame',
                  'distance': 'Mpc (MBHBS reader converts to metres)', 'angles': 'SSB ecliptic radians',
                  't_ref': 'seconds from target_time_origin', 'time_origin': 'absolute orbit seconds',
                  'f_ref': 'Hz; 0 selects BBHx automatic anchor'},
        'target_time_origin': target_time_origin,
        'target': {'waveform': 'BBHx/PhenomHM', 'orbits': 'ESA'},
        'reference_transfer': {
            'phase_rule': 'pi/2 - coa_phase' if thm or native_phase_convention == 'pi-over-2-minus-phase' else 'coa_phase',
            'time_rule': 'tc + native data_time_origin - target_time_origin',
            'approximate': thm,
            'note': ('THM orbital phase at native f_ref and merger tc are approximate BBHx automatic-anchor starting values; '
                     'no exact cross-model phase/time translation or waveform equivalence. Refine both in PE.' if thm else
                     'BBHx anchor retained using caller-declared provider phase convention; no waveform equivalence validation.')},
        'truth': None,
        'capability_note': 'Target describes intent, not an enabled orbit or raw-data pipeline. Keep catalogue truth separate. These are starting values, not catalogue truth or a complete PE configuration. External teams choose priors, sampler, observation window and waveform settings.',
    }
    contents = dict(seed)
    contents['provenance'] = provenance
    out = Path(out)
    serialized = yaml.safe_dump(contents, sort_keys=False)
    logger.debug('Exporting CSV row %s (%s) into %s', row_index, model, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as stream:
        stream.write(serialized)
    return contents
