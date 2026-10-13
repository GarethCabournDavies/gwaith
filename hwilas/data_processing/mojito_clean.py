"""Re-downsample the raw Mojito L1 TDI-2 data (0.25 s) to 4 s with cross-faded blocks.

Run on a cluster compute node (needs ~4 GB RAM), in the hwilas env:
    python make_clean_4s.py RAW_MOJITO_FILE OUTPUT_FILE

Method, per block of KEEP_DAYS core plus PAD_DAYS either side:
  remove linear trend -> Tukey taper over the outer TAPER_DAYS only -> FFT -> keep FLO..FHI Hz with
  smooth cosine roll-offs (FHI well below the 4 s Nyquist of 0.125 Hz, so nothing aliases)
  -> inverse FFT -> take every 16th sample (sample k of the output = raw sample 16k).
Neighbouring blocks use complementary sin^2/cos^2 weights over XFADE_DAYS.
The overlap keeps low-frequency leakage continuous at block boundaries.
Output: datasets LISA_A, LISA_E, LISA_T in Hz (not divided by the laser frequency),
with attrs delta_t=4, epoch=0 (sample 0 = first raw sample), t0_raw (the raw file's
time of its first sample), laser_frequency, and the filter settings.
"""
import h5py
import numpy as np
from scipy.signal.windows import tukey

FS_IN, Q = 4.0, 16
KEEP_DAYS, PAD_DAYS, TAPER_DAYS, XFADE_DAYS = 30.0, 5.0, 1.5, 2.0
FLO, FHI = 2e-5, 0.1          # pass band edges (Hz); cosine roll-off over a factor 2 below FLO and 10% above FHI


def gain(f):
    g = np.ones_like(f)
    lo = f < FLO
    g[lo] = np.where(f[lo] > FLO / 2, 0.5 - 0.5 * np.cos(np.pi * (f[lo] - FLO / 2) / (FLO / 2)), 0.0)
    hi = f > FHI
    g[hi] = np.where(f[hi] < 1.1 * FHI, 0.5 + 0.5 * np.cos(np.pi * (f[hi] - FHI) / (0.1 * FHI)), 0.0)
    return g


def _q(days):
    n = int(days * 86400 * FS_IN)
    return n - n % Q


def clean_block(x, taper):
    """Band-limit and decimate the whole block x (raw samples); taper only the outer `taper` samples."""
    n = len(x)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    x = x * tukey(n, alpha=min(1.0, 2 * taper / n))
    X = np.fft.rfft(x)
    X *= gain(np.fft.rfftfreq(n, 1 / FS_IN))
    return np.fft.irfft(X, n=n)[::Q]


def process_channel(read, n_raw, keep_days=KEEP_DAYS, pad_days=PAD_DAYS, taper_days=TAPER_DAYS,
                    xfade_days=XFADE_DAYS, log=None):
    """read(i0, i1) -> raw samples. Returns the decimated, cross-faded channel (n_raw // Q samples)."""
    keep, pad, taper, xf = _q(keep_days), _q(pad_days), _q(taper_days), _q(xfade_days)
    if pad < xf // 2 + taper or keep <= xf or min(keep, pad, taper, xf) <= 0:
        raise ValueError("Invalid block/taper/cross-fade geometry")
    n_out = n_raw // Q
    out = np.zeros(n_out)
    wsum = np.zeros(n_out)
    nx = xf // Q
    ramp = np.sin(0.5 * np.pi * (np.arange(nx) + 0.5) / nx) ** 2       # 0 -> 1, complementary with 1 - ramp
    for start in range(0, n_out * Q, keep):
        stop = min(start + keep, n_out * Q)
        a, b = max(0, start - pad), min(n_raw, stop + pad)
        x = read(a, b)
        lpad, rpad = start - a, b - stop
        if lpad < pad or rpad < pad:      # pad symmetrically at file edges so the core lands at the same offset
            x = np.pad(x, (pad - lpad, pad - rpad), mode="reflect")
        y = clean_block(x, taper)          # y[k] <-> raw sample start - pad + Q k
        j0, j1 = (start - xf // 2) // Q, (stop + xf // 2) // Q
        w = np.ones(j1 - j0)
        if start > 0:
            w[:nx] = ramp
        if stop < n_out * Q:
            w[-nx:] = 1 - ramp
        lo, hi = max(j0, 0), min(j1, n_out)
        k0 = (pad - xf // 2) // Q + (lo - j0)
        out[lo:hi] += (w * 1.0)[lo - j0:hi - j0] * y[k0:k0 + hi - lo]
        wsum[lo:hi] += w[lo - j0:hi - j0]
        if log:
            log(f"days {start / FS_IN / 86400:7.1f} - {stop / FS_IN / 86400:7.1f} done")
    assert np.allclose(wsum, 1.0), (wsum.min(), wsum.max())
    return out


def prepare_file(raw_path, out_path):
    import mojito
    names = {"LISA_A": "a2", "LISA_E": "e2", "LISA_T": "t2"}
    with mojito.MojitoL1File(raw_path) as f, h5py.File(out_path, "w") as out:
        s = f.tdis.time_sampling
        n_raw = int(s.size)
        if not np.isclose(float(s.fs), FS_IN):
            raise ValueError("This validated recipe requires raw 0.25 s cadence")
        if n_raw < 2:
            raise ValueError("Raw data are too short")
        for c, name in names.items():
            ds = getattr(f.tdis, name)
            out[c] = process_channel(lambda a, b: np.asarray(ds[a:b], float), n_raw,
                                     log=lambda m: print(f"{c}: {m}", flush=True))
        out.attrs.update(delta_t=Q / FS_IN, epoch=0.0, t0_raw=float(s.t(slice(0, 1))[0]),
                         laser_frequency=float(f.laser_frequency), flo=FLO, fhi=FHI,
                         keep_days=KEEP_DAYS, pad_days=PAD_DAYS, taper_days=TAPER_DAYS, xfade_days=XFADE_DAYS,
                         source=raw_path, version=2,
                         method="FFT band-limit (cosine roll-offs) + decimate by 16, overlapping blocks, "
                                "sin^2 cross-fade at block joins")
