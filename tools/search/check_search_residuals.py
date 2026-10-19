"""Check subtraction arithmetic, residual spectra and remaining search candidates."""
import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

from hwilas.search.statistics import read_statistic
import h5py
import numpy as np
from scipy.signal import welch

from hwilas import add_hwilas_common_options, init_logging
from hwilas.search.rounds import validate_grids
from hwilas.types import KeyValueAction

logger = logging.getLogger(__name__)


def outside_signal_windows(times, fitted_times, half_width):
    """Retain events strictly beyond every symmetric fitted-merger window."""
    outside = np.ones(len(times), dtype=bool)
    for tc in fitted_times:
        outside &= abs(times-tc) > half_width
    return outside


def remaining_exposure(intervals, fitted_times, half_width):
    """Union searched intervals and subtract overlapping merger windows once."""
    merged = []
    for start, end in sorted(intervals, key=lambda interval: (interval[0], interval[1])):
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError('Analysis intervals must be finite and increasing')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for tc in fitted_times:
        low, high = tc-half_width, tc+half_width
        pieces = []
        for start, end in merged:
            if high <= start or low >= end:
                pieces.append([start, end])
            else:
                if start < low:
                    pieces.append([start, low])
                if high < end:
                    pieces.append([high, end])
        merged = pieces
    return sum(end-start for start, end in merged)


def check_residuals(data_file, residual_file, signal_files, channels,
                    parameter_file=None, coincidence_file=None, statistic_threshold=40.,
                    chunk_size=65536, compute_psd=False, subtraction_channels=None, end_exclusion=0., signal_exclusion=86400.):
    """Use bounded memory; a subtraction check does not assert noise-only data."""
    if not np.isfinite(signal_exclusion) or signal_exclusion < 0:
        raise ValueError('Signal exclusion must be finite and nonnegative')
    fitted_times = None
    report = {'channels':{}, 'subtraction_consistent':True,
              'diagnostic_statistic_threshold':statistic_threshold}
    spectra = {}
    with ExitStack() as stack:
        data = stack.enter_context(h5py.File(data_file,'r'))
        residual = stack.enter_context(h5py.File(residual_file,'r'))
        if subtraction_channels is None:
            subtraction_channels = json.loads(residual.attrs.get('subtraction_channels', json.dumps(channels)))
        signals = [stack.enter_context(h5py.File(path,'r')) for path in signal_files]
        validate_grids(data,residual,channels)
        for f in signals:
            validate_grids(data,f,subtraction_channels)
        dt = float(data.attrs['delta_t'])
        data_start = float(data.attrs['epoch'])
        data_end = data_start+len(data[channels[0]])*dt-end_exclusion
        for channel in channels:
            n = len(data[channel])
            analysed_n = n-int(np.ceil(end_exclusion/dt))
            if analysed_n<=0:
                raise ValueError("End exclusion leaves no residual samples")
            error, scale, original_power, residual_power = 0.,0.,0.,0.
            finite = True
            psd_sum, blocks = None,0
            for j in range(0,n,chunk_size):
                original = np.asarray(data[channel][j:j+chunk_size],float)
                observed = np.asarray(residual[channel][j:j+chunk_size],float)
                expected = original.copy()
                if channel in subtraction_channels:
                    for signal in signals:
                        expected -= signal[channel][j:j+chunk_size]
                finite &= bool(np.isfinite(observed).all() and np.isfinite(expected).all())
                error = max(error,float(np.max(abs(observed-expected),initial=0.)))
                scale = max(scale,float(np.max(abs(original),initial=0.)),float(np.max(abs(expected),initial=0.)))
                length = max(0,min(len(observed),analysed_n-j))
                original_power += float(np.sum(original[:length]**2))
                residual_power += float(np.sum(observed[:length]**2))
                if compute_psd and length >= min(16384,chunk_size):
                    frequency, psd = welch(observed[:length],fs=1/dt,nperseg=min(16384,chunk_size))
                    psd_sum = psd if psd_sum is None else psd_sum+psd
                    blocks += 1
            consistent = bool(finite and error <= 64*np.finfo(float).eps*max(scale,np.finfo(float).tiny))
            report['subtraction_consistent'] &= consistent
            report['channels'][channel] = dict(samples=n,analysed_samples=analysed_n,finite=finite,
                maximum_subtraction_error=error,original_rms=np.sqrt(original_power/analysed_n),
                residual_rms=np.sqrt(residual_power/analysed_n),subtraction_consistent=consistent)
            if blocks:
                spectra[channel] = (frequency,psd_sum/blocks)
    if parameter_file:
        with h5py.File(parameter_file,'r') as f:
            fitted_times = np.asarray(f['tc'][:], dtype=float)
            if not np.isfinite(fitted_times).all():
                raise ValueError('Fitted merger times must be finite')
            report['fitted_signal_count'] = len(fitted_times)
            if 'search_round' in f:
                rounds,counts = np.unique(f['search_round'][:],return_counts=True)
                report['fitted_signals_per_round'] = {str(int(r)):int(c) for r,c in zip(rounds,counts)}
    if coincidence_file:
        with h5py.File(coincidence_file,'r') as f:
            statistic_values = read_statistic(f)
            times = (f[f'time_{channels[0]}'][:]+f[f'time_{channels[1]}'][:])/2
            if not (np.isfinite(statistic_values).all() and np.isfinite(times).all()):
                raise ValueError('Candidate statistics and times must be finite')
            order = np.argsort(-statistic_values)
            report['residual_coincidence_count'] = len(statistic_values)
            report['residual_coincidences_above_diagnostic_threshold'] = int(np.sum(statistic_values>=statistic_threshold))
            report['maximum_residual_statistic'] = float(np.max(statistic_values,initial=0.))
            report['loudest_residual_candidates'] = [dict(statistic=float(statistic_values[i]),
                time=float((f[f'time_{channels[0]}'][i]+f[f'time_{channels[1]}'][i])/2),
                template_id=int(f['template_id'][i])) for i in order[:10]]
            if fitted_times is not None:
                outside = outside_signal_windows(times, fitted_times, signal_exclusion)
                intervals = np.asarray(f.attrs.get('analysis_intervals', [[data_start, data_end]]), dtype=float)
                if intervals.ndim != 2 or intervals.shape[1] != 2:
                    raise ValueError('Analysis intervals must have shape (n, 2)')
                if 'analysis_intervals' not in f.attrs:
                    logger.warning('Candidate analysis intervals unavailable; using data span minus end exclusion for event rates')
                exposure = remaining_exposure(intervals, fitted_times, signal_exclusion)
                report['residual_search_exposure_seconds_outside_signal_windows'] = exposure
                report['residual_search_exposure_source'] = ('candidate analysis_intervals'
                    if 'analysis_intervals' in f.attrs else 'data span minus end exclusion')
                remaining = statistic_values[outside]
                thresholds, counts = np.unique(remaining, return_counts=True)
                report['signal_exclusion_seconds'] = float(signal_exclusion)
                report['residual_coincidence_count_outside_signal_windows'] = len(remaining)
                report['maximum_residual_statistic_outside_signal_windows'] = (
                    float(np.max(remaining)) if len(remaining) else None)
                cumulative = np.cumsum(counts[::-1])[::-1]
                report['residual_statistic_survival_outside_signal_windows'] = dict(
                    statistic=thresholds.tolist(), count_at_or_above=cumulative.tolist(),
                    rate_hz_at_or_above=(cumulative/exposure).tolist() if exposure > 0 else [])
                logger.debug('Retained %d of %d candidates outside fitted signal windows',
                             len(remaining), len(times))
    return report,spectra


def plot_statistic_survival(report, output_file):
    """Plot cumulative event counts, including a visible marker for a lone event."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    survival = report['residual_statistic_survival_outside_signal_windows']
    fig,ax = plt.subplots(figsize=(10,6))
    ax.step(survival['statistic'],survival['rate_hz_at_or_above'],where='pre',marker='o',label='Residual events',zorder=3)
    ax.legend()
    if not survival['statistic']:
        ax.text(.5,.5,'No candidates outside signal windows',ha='center',transform=ax.transAxes)
    ax.set_xlabel('Ranking statistic')
    ax.set_ylabel('Rate with statistic ≥ threshold (Hz)')
    ax.set_title('Residual candidates')
    ax.set_yscale('log')
    if not survival['statistic']:
        ax.set_ylim(1e-9, 1.)
    Path(output_file).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_file,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    parser.add_argument('--data-file',required=True)
    parser.add_argument('--residual-file',required=True)
    parser.add_argument('--signal-files',nargs='+',required=True)
    parser.add_argument('--parameter-file')
    parser.add_argument('--candidate-file', '--coincidence-file', dest='candidate_file')
    parser.add_argument('--channels',nargs='+',default=['LISA_A','LISA_E','LISA_T'])
    parser.add_argument('--end-exclusion',type=float,default=0.,help='Seconds omitted from residual RMS/PSD diagnostics')
    parser.add_argument('--subtraction-channels',nargs='+',
                        help='Channels whose models were subtracted; other channels must be unchanged')
    parser.add_argument('--statistic-threshold', '--snr-threshold', dest='statistic_threshold',type=float,default=40.,help='Diagnostic residual threshold, not an optimisation selection cut')
    parser.add_argument('--signal-exclusion',type=float,default=86400.,
                        help='Seconds excluded either side of fitted tc (default: one day)')
    parser.add_argument('--output-statistic-plot',help='Cumulative remaining statistics; requires parameter and candidate files')
    parser.add_argument('--output-file',required=True,help='JSON residual report')
    parser.add_argument('--output-plot',help='Residual PSD comparison filename')
    parser.add_argument('--psd-files',nargs='+',action=KeyValueAction)
    args = parser.parse_args()
    if args.output_statistic_plot and not (args.parameter_file and args.candidate_file):
        parser.error('--output-statistic-plot requires --parameter-file and --candidate-file')
    init_logging(args.verbose)
    report,spectra = check_residuals(args.data_file,args.residual_file,args.signal_files,
        args.channels,args.parameter_file,args.candidate_file,args.statistic_threshold,
        compute_psd=bool(args.output_plot),subtraction_channels=args.subtraction_channels,end_exclusion=args.end_exclusion,signal_exclusion=args.signal_exclusion)
    Path(args.output_file).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output_file,'w') as f:
        json.dump(report,f,indent=2)
    if args.output_plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,ax = plt.subplots(figsize=(10,6))
        for i,(channel,(frequency,psd)) in enumerate(spectra.items()):
            mask = (frequency>0)&(psd>0)
            ax.loglog(frequency[mask],psd[mask],color=f'C{i}',label=f'{channel} residual')
            if args.psd_files and channel in args.psd_files:
                reference = np.loadtxt(args.psd_files[channel])
                mask = (reference[:,0]>0)&(reference[:,1]>0)
                ax.loglog(reference[mask,0],reference[mask,1],color=f'C{i}',linestyle='--',alpha=.7,label=f'{channel} search PSD')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.legend()
        Path(args.output_plot).parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.output_plot,bbox_inches='tight')
        plt.close(fig)
    if args.output_statistic_plot:
        plot_statistic_survival(report,args.output_statistic_plot)
    if 'maximum_residual_statistic_outside_signal_windows' in report:
        logger.info('Highest residual statistic outside ±%g seconds of fitted mergers: %s',
                    args.signal_exclusion,report['maximum_residual_statistic_outside_signal_windows'])
    logger.info('Residual subtraction consistent: %s', report['subtraction_consistent'])
    if not report['subtraction_consistent']:
        raise SystemExit('Residual is nonfinite or differs from original minus all models')


if __name__ == '__main__':
    main()
