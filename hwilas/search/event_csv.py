"""Consistent event CSV layout for candidates and fitted multi-round catalogues."""
import csv
import logging
import h5py
import numpy as np
from hwilas.search.statistics import canonical_statistics
from hwilas.conversions import chi_eff_from_component_spins, mchirp_eta_from_mass1_mass2

logger = logging.getLogger(__name__)

LEADING_COLUMNS = ('event_id', 'time_s', 'time_days', 'statistic', 'mchirp', 'eta', 'chieff')
PHYSICAL_COLUMNS = ('mass1', 'mass2', 'spin1z', 'spin2z', 'distance',
                    'eclipticlongitude', 'eclipticlatitude', 'inclination',
                    'polarization', 'coa_phase', 'f_lower')


def write_event_csv(path, columns):
    """Put physical seed parameters first and append diagnostics consistently."""
    n = len(columns['statistic'])
    for key in LEADING_COLUMNS:
        columns.setdefault(key, np.full(n, np.nan))
    keys = [*LEADING_COLUMNS, *(k for k in PHYSICAL_COLUMNS if k in columns),
            *sorted(set(columns)-set(LEADING_COLUMNS)-set(PHYSICAL_COLUMNS))]
    order = np.argsort(-columns['statistic'], kind='stable')
    with open(path, 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(keys)
        for index in order:
            writer.writerow([columns[key][index] for key in keys])


def write_candidate_csv(path, rows, bank_file=None, statistic='same-time A/E quadrature SNR'):
    """Export candidates in decreasing statistic order, marking template-derived parameters."""
    from hwilas.conversions import mchirp_eta_from_mass1_mass2, chi_eff_from_component_spins
    rows = canonical_statistics(rows)
    channels = [key[10:] for key in rows if key.startswith('statistic_')]
    times = rows.get('end_time')
    if times is None:
        times = np.mean([rows[f'time_{c}'] for c in channels], axis=0)
    statistics = rows.get('statistic')
    if statistics is None:
        statistics = np.sqrt(sum(rows[f'statistic_{c}']**2 for c in channels))
    columns = dict(event_id=rows.get('combined_event_index', np.arange(len(statistics))),
        time_s=times, time_days=times/86400., statistic=statistics, template_id=rows['template_id'])
    for key in ('combined_event_index', 'search_round', 'round_event_index', 'close_other_round',
                'nearest_other_round_index', 'nearest_other_round_separation_seconds'):
        if key in rows:
            columns[key] = rows[key]
    for c in channels:
        columns[f'statistic_{c}'] = rows[f'statistic_{c}']
        columns[f'time_{c}_seconds'] = rows[f'time_{c}']
        if f'complex_snr_{c}' in rows:
            columns[f'statistic_phase_{c}_rad'] = np.angle(rows[f'complex_snr_{c}'])
    if bank_file:
        ids = np.asarray(rows['template_id'], dtype=int)
        with h5py.File(bank_file) as bank:
            if np.any(ids < 0) or np.any(ids >= len(bank['mass1'])):
                raise ValueError('Candidate template ID is outside the supplied bank')
            for key in PHYSICAL_COLUMNS:
                if key in bank:
                    columns[key] = bank[key][:][ids]
            m1, m2 = columns['mass1'], columns['mass2']
            columns['mchirp'], columns['eta'] = mchirp_eta_from_mass1_mass2(m1, m2)
            columns['chieff'] = chi_eff_from_component_spins(m1, m2,
                columns['spin1z'], columns['spin2z'])
    columns['parameter_source'] = np.full(len(statistics), 'template' if bank_file else 'unavailable')
    columns['statistic_name'] = np.full(len(statistics), statistic) if isinstance(statistic, str) else np.asarray(statistic)
    write_event_csv(path, columns)
    logger.info('Exported %d clustered candidates to %s', len(statistics), path)


def write_optimised_csv(path, rows, optimizer='unknown'):
    """Preserve fitted event columns, units, original row identity and fit diagnostics."""
    names = dict(tc='time_s', optimised_stat='statistic', initial_fit_snr='initial_fit_stat')
    columns = {names.get(k, k): v for k, v in rows.items()}
    n = len(rows['tc'])
    columns.setdefault('statistic', np.full(n, np.nan))
    columns['event_id'] = rows.get('combined_event_index', rows.get('global_event_index', np.arange(n)))
    columns['time_days'] = rows['tc']/86400.
    if 'mass1' in rows and 'mass2' in rows:
        mc, eta = mchirp_eta_from_mass1_mass2(rows['mass1'], rows['mass2'])
        columns.setdefault('mchirp', mc)
        columns.setdefault('eta', eta)
        if 'spin1z' in rows and 'spin2z' in rows:
            columns['chieff'] = chi_eff_from_component_spins(rows['mass1'], rows['mass2'], rows['spin1z'], rows['spin2z'])
    columns['optimizer'] = np.full(n, optimizer) if isinstance(optimizer, str) else np.asarray(optimizer)
    columns['parameter_source'] = np.full(n, 'optimised')
    write_event_csv(path, columns)
    logger.info('Exported %d fitted events to %s', len(rows['tc']), path)


