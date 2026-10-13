"""Local event-window comparisons for noisy data, injections and fitted models."""
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

from gwaith.types import TimeSeries
from hwilas.conversions import mchirp_eta_from_mass1_mass2


def load_parameters(path):
    """Load one-dimensional event columns and per-round waveform metadata."""
    with h5py.File(path,'r') as f:
        rows = [{key:f[key][i].item() for key in f
                 if isinstance(f[key],h5py.Dataset) and f[key].ndim==1}
                for i in range(len(f['tc']))]
        attrs = dict(f.attrs)
        metadata = [dict(f['round_metadata'][str(row['search_round'])].attrs)
                    if 'round_metadata' in f else attrs.copy() for row in rows]
    return rows,metadata


def load_catalogue(path,time_offset=800.):
    """Read standard truth columns, keeping source IDs and catalogue conventions."""
    with h5py.File(path,'r') as f:
        rows=[]
        for i in range(len(f['tc'])):
            row={key:float(f[key][i]) for key in ['mass1','mass2','distance',
                 'eclipticlongitude','eclipticlatitude','inclination','polarization']}
            row.update(source_id=int(f['ID'][i]),tc=float(f['tc'][i])-time_offset,
                       catalogue_tc=float(f['tc'][i]),spin1z=float(f['PrimarySpinCompZ'][i]),
                       spin2z=float(f['SecondarySpinCompZ'][i]),
                       coa_phase=float(f['PhaseReferenceSourceFrame'][i]),
                       expected_snr=float(f['EstimatedSNR'][i]))
            row['mchirp'],row['eta']=mchirp_eta_from_mass1_mass2(row['mass1'],row['mass2'])
            rows.append(row)
    return sorted(rows,key=lambda row:row['tc'])


def match_catalogue_to_fits(catalogue,fits,tolerance=3600.):
    """Assign fits uniquely by merger time; leave nearby unfitted sources unmatched."""
    matches={row['source_id']:None for row in catalogue}
    if not catalogue or not fits:
        return matches
    costs=abs(np.array([row['tc'] for row in catalogue])[:,None]
              -np.array([row['tc'] for row in fits])[None,:])
    assignment_costs=np.where(costs<=tolerance,costs,
                              (len(catalogue)+len(fits)+1)*max(tolerance,1.))
    truth_ids,fit_ids=linear_sum_assignment(assignment_costs)
    for i,j in zip(truth_ids,fit_ids):
        if costs[i,j]<=tolerance:
            matches[catalogue[i]['source_id']]=int(j)
    return matches


def raw_time_origin(paths, reference_delta_t=.25):
    """Find the common absolute time origin from the nominal-cadence slices."""
    origins = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            if 'fs' in f.attrs and 't' in f and np.isclose(1/float(f.attrs['fs']), reference_delta_t):
                origins.append(float(f['t'][0])-float(f.attrs['i0'])/float(f.attrs['fs']))
    if not origins:
        raise ValueError('No nominal-cadence timestamped slices establish the Mojito time origin')
    if not np.allclose(origins, origins[0], rtol=0., atol=1e-6):
        raise ValueError('Nominal-cadence slices disagree on the Mojito time origin')
    return origins[0]


def read_window(path,channels,epoch,n,dt,laser_frequency,fractional_default=True,
                epoch_shift=0.,raw_bandlimit=False,raw_time_reference=None):
    """Read only local samples and align them to the target fractional-frequency grid.

    Standard processed HDF5 files use epoch/delta_t and LISA_A/E/T. Raw
    source slices use fs and a2/e2/t2 in laser-frequency units. Stored
    timestamps are aligned to a shared raw_time_reference, retaining source offsets.
    If omitted, infer that reference from sibling nominal-cadence Mojito slices.
    Unsupported times are NaN, rather than invented zero-valued data.
    """
    from hwilas.data_processing.mojito_clean import gain
    times=epoch+np.arange(n)*dt
    result={}
    with h5py.File(path,'r') as f:
        raw_dt=float(f.attrs['delta_t']) if 'delta_t' in f.attrs else 1/float(f.attrs['fs'])
        start=float(f.attrs.get('epoch',float(f.attrs.get('i0',0))*raw_dt))
        if 'epoch' not in f.attrs and 't' in f:
            if raw_time_reference is None:
                # Sibling nominal-cadence slices establish one origin for all sources.
                reference_paths = sorted(Path(path).parent.glob('mbhb_source*_slice.h5'))
                try:
                    raw_time_reference = raw_time_origin(reference_paths)
                except ValueError as error:
                    raise ValueError('Timestamped raw slices require a shared raw_time_reference; '
                                     f'automatic sibling lookup failed: {error}') from error
            start=float(f['t'][0])-raw_time_reference
            if len(f['t']) > 1 and not np.isclose(float(f['t'][1]-f['t'][0]),raw_dt):
                raise ValueError('Raw timestamps disagree with the recorded cadence')
        start+=epoch_shift
        fractional=bool(f.attrs.get('divide_by_laser_frequency',fractional_default))
        laser=float(f.attrs.get('laser_frequency',laser_frequency))
        if raw_dt<=0 or dt<raw_dt:
            raise ValueError('Input cadence must be positive and no coarser than the requested cadence')
        for c in channels:
            keys=[c,{'LISA_A':'a2','LISA_E':'e2','LISA_T':'t2'}[c]]
            key=next((key for key in keys if key in f),None)
            if key is None:
                raise KeyError(f'{path}: missing channel {c}')
            ds=f[key]
            # Read context for anti-alias filtering; raw band limiting needs longer context.
            padding=1.5*86400 if raw_bandlimit and dt>raw_dt else 64*dt
            j0=max(0,int(np.floor((times[0]-padding-start)/raw_dt)))
            j1=min(len(ds),int(np.ceil((times[-1]+padding-start)/raw_dt))+1)
            aligned=np.full(n,np.nan)
            if j1>j0:
                x=np.asarray(ds[j0:j1],float)
                if not fractional:
                    x/=laser
                if raw_bandlimit and dt>raw_dt:
                    frequencies=np.fft.rfftfreq(len(x),raw_dt)
                    x=np.fft.irfft(np.fft.rfft(x)*gain(frequencies),n=len(x))
                ratio=Fraction(raw_dt/dt).limit_denominator(100000)
                y=resample_poly(x,ratio.numerator,ratio.denominator) if dt>raw_dt else x
                origin=start+j0*raw_dt
                positions=(times-origin)/dt
                # Apply the fractional grid offset in Fourier space, avoiding linear-interpolation attenuation.
                nearest=np.rint(positions).astype(int)
                offset=float(positions[0]-nearest[0])
                if abs(offset)>1e-9:
                    extra=128
                    padded=np.pad(y,(extra,extra))
                    freq=np.fft.rfftfreq(len(padded),dt)
                    y=np.fft.irfft(np.fft.rfft(padded)*np.exp(2j*np.pi*freq*offset*dt),n=len(padded))[extra:extra+len(y)]
                valid=(nearest>=0)&(nearest<len(y))&(times>=start)&(times<start+len(ds)*raw_dt)
                aligned[valid]=y[nearest[valid]]
            result[c]=aligned
    return result


def waveform_window(params,metadata,channels,epoch,n,dt,laser_scale=1.):
    """Regenerate a physical model on its saved FFT grid, then crop to the plot grid."""
    from tools.search.optimise_snr_peaks import build_aligned_waveform, PARAMETERS, DEFAULT_HM_MODE_ARRAY
    pre_days=float(metadata.get('pre_merger_days',1.))
    if pre_days!=1. or float(metadata.get('taper_seconds',3600.))!=3600.:
        raise ValueError('The current optimiser generator requires one-day support and a one-hour taper')
    modes=np.asarray(metadata.get('mode_array',DEFAULT_HM_MODE_ARRAY))
    if not np.array_equal(modes,np.asarray(DEFAULT_HM_MODE_ARRAY)):
        raise ValueError('Saved mode array differs from the current optimiser generator')
    if metadata.get('approximant','IMRPhenomHM')!='IMRPhenomHM':
        raise ValueError('This notebook generator currently reproduces IMRPhenomHM fits')
    if not np.isclose(float(params.get('amplitude_phase',0.)),0.):
        raise ValueError('Regenerate the legacy complex-amplitude fit with the physical-distance optimiser')
    start=float(params.get('optimisation_window_start',np.floor((params['tc']-2*86400)/dt)*dt))
    end=float(params.get('optimisation_window_end',start+4*86400))
    length=round((end-start)/dt)
    size=next_fast_len(length)
    if size%2:
        size=next_fast_len(size+1)
    dummy={c:TimeSeries(np.zeros(size),epoch=start,delta_t=dt) for c in channels}
    shared=dict(f_lower=float(metadata.get('f_lower_floor',3e-5)),
                f_upper=float(metadata.get('f_upper',.09)),tdi=str(metadata.get('tdi_generation','2')),
                distance=params['distance'],template_scale=laser_scale)
    vector=np.array([params[k] for k in PARAMETERS])
    fd=build_aligned_waveform(vector,dummy,shared,True,waveform_channels=channels)
    positions=np.rint((epoch+np.arange(n)*dt-start)/dt).astype(int)
    if not np.allclose((epoch-start)/dt,round((epoch-start)/dt)):
        raise ValueError('Model and display grids must have the same sample origin')
    valid=(positions>=0)&(positions<length)
    result={c:np.zeros(n) for c in channels}
    for c in channels:
        signal=np.fft.irfft(fd[c].data,n=size)/dt
        result[c][valid]=signal[positions[valid]]
    return result


def noiseless_path(source_id,combined_file,directory,pattern):
    """Prefer a configured combined injection file, otherwise a source-specific slice."""
    path=Path(combined_file) if combined_file is not None else Path(directory)/pattern.format(source_id=source_id)
    return path if path.is_file() else None


def noiseless_match(injection, model, dt, psds, f_lower, f_upper):
    """Return signed, normalised PSD-weighted overlap on the supplied time window.

    Keep the physical fitted time and phase. Each PSD is a frequency/value pair;
    channels with missing samples cannot establish a match and return NaN.
    """
    cross = norm_injection = norm_model = 0.
    for channel, (frequencies, values) in psds.items():
        x, y = np.asarray(injection[channel]), np.asarray(model[channel])
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError('Injection and model must share a one-dimensional time grid')
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return np.nan
        frequencies, values = np.asarray(frequencies), np.asarray(values)
        valid = (frequencies > 0) & (values > 0) & np.isfinite(values)
        frequencies, values = frequencies[valid], values[valid]
        if len(frequencies) < 2 or np.any(np.diff(frequencies) <= 0):
            raise ValueError('PSD requires increasing positive frequencies and positive values')
        freq = np.fft.rfftfreq(len(x), dt)
        band = (freq >= f_lower) & (freq <= f_upper) & (freq > 0)
        band &= (freq >= frequencies[0]) & (freq <= frequencies[-1])
        weight = 1 / np.exp(np.interp(np.log(freq[band]), np.log(frequencies), np.log(values)))
        # The common FFT normalisation cancels in the normalised overlap.
        a, b = np.fft.rfft(x)[band], np.fft.rfft(y)[band]
        if len(x) % 2 == 0 and band[-1]:
            weight[-1] *= .5
        cross += np.sum((a.conj()*b).real*weight)
        norm_injection += np.sum(abs(a)**2*weight)
        norm_model += np.sum(abs(b)**2*weight)
    if norm_injection <= 0 or norm_model <= 0:
        return np.nan
    return float(np.clip(cross / np.sqrt(norm_injection*norm_model), -1., 1.))
