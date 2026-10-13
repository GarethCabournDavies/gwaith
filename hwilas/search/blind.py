"""Fixed-epoch, local-PSD blind search without catalogue access.

Waveforms are generated once per template and reused across overlapping chunks.
Prepared data are memory mapped by spawned workers. A manifest prevents mixing
triggers from different data, banks, settings, or template selections.
"""
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.fft import ifft, next_fast_len
from scipy.signal import detrend, find_peaks, welch
from scipy.signal.windows import tukey
from hwilas.template_banks.io import load_bank
from hwilas.waveforms.short_fastbhb import CHANNELS, short_template, template_geometry
from hwilas.template_banks.frequency import duration_from_frequency

logger = logging.getLogger(__name__)
_G = {}


def selected_channels(cfg):
    channel = cfg["search"].get("channel")
    if channel is None:
        return CHANNELS
    if channel not in CHANNELS:
        raise ValueError(f"Unsupported search channel: {channel}")
    return (channel,)


def geometry(cfg, n_samples=None):
    w, s = cfg["waveform"], cfg["search"]
    dt = w["delta_t"]
    requested = n_samples if s.get("whole_file", False) else round(s.get("chunk_days", 30.)*86400/dt)
    if requested is None:
        raise ValueError("Whole-file geometry requires the input sample count")
    n = next_fast_len(requested)
    edge = round(s.get("edge_days", 1.5)*86400/dt)
    step = n-2*edge
    if step <= 0 or edge < round(1.5*86400/dt) or n < round(2*86400/dt):
        raise ValueError("Chunks need a nonempty core and at least 1.5-day edge padding")
    return n, edge, step


def signature(cfg):
    payload = {"waveform": cfg["waveform"], "search": cfg["search"], "version": 1}
    for key in ("data_file", "bank_file", "psd_A", "psd_E"):
        if key not in cfg["search"]:
            continue
        p = Path(cfg["search"][key])
        st = p.stat()
        payload[key] = dict(path=str(p.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare(cfg):
    s, w = cfg["search"], cfg["waveform"]
    channels = selected_channels(cfg)
    out = Path(s["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("Prepared output exists; use another output_dir for changed inputs")
    t0 = time.monotonic()
    dt = w["delta_t"]
    with h5py.File(s["data_file"]) as f:
        if not np.isclose(f.attrs["delta_t"], dt):
            raise ValueError("Data cadence differs from waveform configuration")
        epoch = float(f.attrs.get("epoch", 0.))
        fractional = bool(f.attrs.get("divide_by_laser_frequency", False))
        unit_scale = float(f.attrs["laser_frequency"]) if fractional else 1.
        if not np.isfinite(unit_scale) or unit_scale <= 0:
            raise ValueError("Invalid laser frequency for fractional-frequency data")
        ntot = len(f[channels[0]])
        n, edge, step = geometry(cfg, ntot)
        whole_file = s.get("whole_file", False)
        if any(len(f[c]) != ntot for c in channels) or ntot <= 2*edge or (not whole_file and ntot < n):
            raise ValueError("Data must contain equal A/E arrays and at least one chunk")
        if "laser_frequency" in f.attrs and not np.isclose(f.attrs["laser_frequency"], w["laser_frequency"]):
            raise ValueError("Template laser frequency differs from data units")
        starts = [0] if whole_file else list(range(0, ntot-n+1, step))
        if not whole_file and starts[-1]+n < ntot:
            starts.append(ntot-n)
        shape = (len(starts), len(channels), n//2+1)
        Q = np.lib.format.open_memmap(out/"Q.npy", mode="w+", shape=shape, dtype=np.complex64)
        IS = np.lib.format.open_memmap(out/"invS.npy", mode="w+", shape=shape, dtype=np.float32)
        freq = np.fft.rfftfreq(n, dt)
        band = (freq >= w["f_lower"]) & (freq <= w["f_upper"])
        win = tukey(n, alpha=min(1., 2*.25*86400/dt/n))
        gate_days = s.get("gate_days", 1.5)
        for i, start in enumerate(starts):
            t = (start+np.arange(n))*dt/86400-gate_days
            gate = .5-.5*np.cos(np.pi*np.clip(t/.25, 0, 1)) if gate_days > 0 else 1.
            for j, c in enumerate(channels):
                d = np.asarray(f[c][start:start+n], float)*unit_scale
                if not np.isfinite(d).all():
                    raise ValueError(f"Nonfinite data in {c} chunk {i}")
                psd_path = s.get("psd_" + c[-1])
                if psd_path:
                    from hwilas.psd.read import from_txt
                    S = from_txt(psd_path, len(freq), 1/(n*dt), 1/(n*dt),
                                 is_asd_file=False).data * unit_scale**2
                    if len(S) != len(freq):
                        raise ValueError("Search PSD does not cover the FFT grid")
                    S[0] = np.inf
                else:
                    fw, pw = welch(d, fs=1/dt, nperseg=min(len(d), round(2*86400/dt)), average="median")
                    if np.any(pw[1:] <= 0):
                        raise ValueError("Local PSD has nonpositive bins")
                    S = np.exp(np.interp(np.log(np.maximum(freq, 1e-20)), np.log(fw[1:]), np.log(pw[1:])))
                # Pad only the FFT buffer; preserve the original data duration.
                windowed = np.pad(detrend(d), (0, n-len(d))) * win * gate
                inv = np.where(band, 1/S, 0.)
                Q[i, j] = dt*np.fft.rfft(windowed)*inv
                IS[i, j] = inv
            logger.info("Prepared chunk %d/%d", i+1, len(starts))
        Q.flush(); IS.flush()
    np.save(out/"starts.npy", starts)
    manifest = dict(signature=signature(cfg), epoch=epoch, n=n, edge=edge, dt=dt,
                    chunks=len(starts), samples=ntot, channels=list(channels), seconds=time.monotonic()-t0,
                    input_units="fractional_frequency" if fractional else "Hz",
                    input_to_Hz_scale=unit_scale,
                    first_valid_time=epoch+edge*dt, last_valid_time=epoch+(ntot-edge)*dt)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def init_worker(cfg):
    out = Path(cfg["search"]["output_dir"])
    _G.update(cfg=cfg, Q=np.load(out/"Q.npy", mmap_mode="r"),
              IS=np.load(out/"invS.npy", mmap_mode="r"), starts=np.load(out/"starts.npy"),
              meta=json.loads((out/"manifest.json").read_text()))


def search_template(item):
    i, p = item
    cfg, m = _G["cfg"], _G["meta"]
    n, edge, dt = m["n"], m["edge"], m["dt"]
    settings = cfg['waveform']
    if 'pre_merger_seconds' in p:
        settings = {**settings, 'f_lower': p['f_lower'],
                    'pre_merger_seconds': duration_from_frequency(p, p['f_lower'])+settings['taper_seconds'],
                    'post_merger_seconds': 86400.}
    length, merger = template_geometry(p, settings)
    if length > n:
        raise ValueError('Template exceeds search chunk; increase chunk_days and edge_days')
    if 'pre_merger_seconds' in settings and not cfg['search'].get('whole_file', False) and edge < length:
        raise ValueError('Template exceeds chunk edge support; increase edge_days')
    tm = short_template(p, settings, modes=[(2, 2)])
    H = [dt*np.fft.rfft(np.pad(tm[c], (0, n-len(tm[c])))) for c in selected_channels(cfg)]
    merger_idx = (merger+np.arange(n)) % n
    valid_end = m["samples"]-edge if cfg["search"].get("whole_file", False) else n-edge
    if 'pre_merger_seconds' in settings:
        edge = max(edge, length)
        valid_end = min(valid_end, m['samples']-length if cfg['search'].get('whole_file', False) else n-length)
    valid = (merger_idx >= edge) & (merger_idx < valid_end)
    rows = []
    df = 1/(n*dt)
    for ci, start in enumerate(_G["starts"]):
        net2 = np.zeros(n)
        for j in range(len(H)):
            sig2 = 4*df*np.sum(abs(H[j])**2*_G["IS"][ci, j])
            if not np.isfinite(sig2) or sig2 <= 0:
                raise ValueError(f"Invalid norm for template {i}, chunk {ci}")
            z = np.zeros(n, np.complex64)
            z[:len(H[j])] = _G["Q"][ci, j]*np.conj(H[j])
            net2 += abs(ifft(z, workers=1)*n*4*df)**2/sig2
        net = np.sqrt(net2)
        pk, _ = find_peaks(np.where(valid, net, 0.), height=cfg["search"].get("statistic_threshold", cfg["search"].get("snr_threshold", 8.)),
                           distance=max(1, round(cfg["search"].get("peak_separation_seconds", 3600.)/dt)))
        for k in pk:
            rows.append((i, m["epoch"]+(start+merger_idx[k])*dt, net[k]))
    return i, np.asarray(rows, float).reshape(-1, 3)


def checked_manifest(cfg):
    p = Path(cfg["search"]["output_dir"])/"manifest.json"
    m = json.loads(p.read_text())
    if m["signature"] != signature(cfg):
        raise ValueError("Inputs/settings changed since prepare; use a new output_dir")
    return m


def search(cfg, nproc=4, max_templates=None):
    if nproc < 1:
        raise ValueError("nproc must be positive")
    checked_manifest(cfg)
    bank = load_bank(cfg["search"]["bank_file"])
    ids = list(range(len(bank)))
    if max_templates is not None:
        if max_templates < 1:
            raise ValueError("max_templates must be positive")
        ids = ids[:max_templates]
    out = Path(cfg["search"]["output_dir"])
    (out/"triggers").mkdir(exist_ok=True)
    todo = [(i, bank[i]) for i in ids if not (out/"triggers"/f"t{i}.npy").exists()]
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(nproc, initializer=init_worker, initargs=(cfg,)) as pool:
        for k, (i, rows) in enumerate(pool.imap_unordered(search_template, todo, chunksize=1)):
            path = out/"triggers"/f"t{i}.npy"
            with path.with_suffix(".tmp").open("wb") as f:
                np.save(f, rows)
            path.with_suffix(".tmp").replace(path)
            if (k+1) % 10 == 0 or k+1 == len(todo):
                logger.info("%d/%d templates, %.1fs", k+1, len(todo), time.monotonic()-t0)
    result = dict(templates_completed=len(ids), templates_in_bank=len(bank),
                  newly_processed=len(todo), seconds=time.monotonic()-t0, nproc=nproc)
    if todo:
        result["estimated_full_search_seconds"] = result["seconds"]*len(bank)/len(todo)
    (out/"timing.json").write_text(json.dumps(result, indent=2))
    return result


def cluster(rows, window):
    kept = []
    for i in np.argsort(-rows[:, 2]):
        if all(abs(rows[i, 1]-r[1]) >= window for r in kept):
            kept.append(rows[i])
    return np.asarray(kept, float).reshape(-1, 3)


def collect(cfg, allow_partial=False):
    m = checked_manifest(cfg)
    s = cfg["search"]
    out = Path(s["output_dir"])
    bank = load_bank(s["bank_file"])
    missing = [i for i in range(len(bank)) if not (out/"triggers"/f"t{i}.npy").exists()]
    if missing and not allow_partial:
        raise ValueError(f"Search incomplete: {len(missing)} templates missing")
    parts = [np.load(out/"triggers"/f"t{i}.npy") for i in range(len(bank)) if i not in missing]
    rows = np.concatenate(parts) if parts else np.empty((0, 3))
    if s.get("channel"):
        # Preserve per-template triggers for manual merging and coincidence.
        order = np.lexsort((rows[:, 1], rows[:, 0]))
        sorted_rows = rows[order]
        with h5py.File(out/"triggers.hdf", "w") as f:
            f.attrs.update(channel=s["channel"], complete=not missing,
                           epoch=m["epoch"], delta_t=m["dt"], bank_file=s["bank_file"])
            f["template_id"] = sorted_rows[:, 0].astype(np.int32)
            f["end_time"] = sorted_rows[:, 1]
            f["statistic"] = sorted_rows[:, 2]
    cand = cluster(rows, s.get("cluster_seconds", 4*3600.))
    # Keep raw clustered candidates: a ratio veto can hide real nearby signals.
    final = cand
    if s.get("apply_ratio_veto", False):
        ratio, days = s.get("veto_ratio", .25), s.get("veto_days", 3.)
        if not 0 < ratio < 1 or days <= 0:
            raise ValueError("Invalid ratio-veto settings")
        final = np.asarray([c for c in cand if not np.any((abs(cand[:, 1]-c[1]) < days*86400) &
                         (cand[:, 2] > c[2]/ratio))]).reshape(-1, 3)
    def records(a):
        return [dict(template=int(r[0]), t=float(r[1]), day=float(r[1]/86400), statistic=float(r[2])) for r in a]
    result = dict(n_triggers=len(rows), n_candidates=len(cand), candidates=records(cand),
                  final_candidates=records(final), complete=not missing, missing_templates=missing,
                  data=s["data_file"], manifest=m)
    (out/"candidates.json").write_text(json.dumps(result, indent=2))
    return result
