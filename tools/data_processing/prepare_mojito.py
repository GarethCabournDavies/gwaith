"""Prepare cross-faded 4 s Mojito data and bank PSDs for transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
from scipy.signal import welch
from hwilas.data_processing.mojito_clean import prepare_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--catalogue", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    clean = out/"mojito_clean_4s_aet_v2.h5"
    if clean.exists():
        raise FileExistsError(f"Refusing to overwrite {clean}")
    # Inspect both inputs before starting the expensive processing stage.
    with h5py.File(a.catalogue) as f:
        catalogue_summary = {}
        def record(name, obj):
            if isinstance(obj, h5py.Dataset):
                catalogue_summary[name] = dict(shape=obj.shape, dtype=str(obj.dtype))
        f.visititems(record)
    prepare_file(a.raw, str(clean))
    with h5py.File(clean) as f:
        for c in ("LISA_A", "LISA_E"):
            # Median across per-chunk median Welch estimates limits signal contamination.
            segment = round(30*86400/4)
            spectra = []
            for start in range(round(1.75*86400/4), len(f[c])-segment+1, segment):
                freq, psd = welch(f[c][start:start+segment], fs=.25,
                                  nperseg=round(2*86400/4), average="median")
                spectra.append(psd)
            if not spectra:
                raise ValueError("Need at least 32 days of raw data for bank PSD preparation")
            np.savetxt(out/f"{c[-1]}_median_psd_Hz.txt", np.column_stack([freq[1:], np.median(spectra, axis=0)[1:]]),
                       header="frequency_Hz PSD_Hz_squared_per_Hz")
    cat = out/("catalogue"+Path(a.catalogue).suffix)
    shutil.copy2(a.catalogue, cat)
    (out/"catalogue_layout.json").write_text(json.dumps(catalogue_summary, indent=2))
    manifest = dict(raw=str(Path(a.raw).resolve()), catalogue=str(Path(a.catalogue).resolve()),
                    catalogue_time_origin_offset_seconds=800., method="FFT band limiting with cross-fades, raw 0.25s -> 4s",
                    files={})
    for path in (clean, cat, out/"A_median_psd_Hz.txt", out/"E_median_psd_Hz.txt"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b""):
                h.update(chunk)
        manifest["files"][path.name] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
    (out/"transfer_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out/"SHA256SUMS").open("w") as f:
        for name, meta in manifest["files"].items():
            f.write(f"{meta['sha256']}  {name}\n")
    print(f"Ready to transfer: {out}", flush=True)


if __name__ == "__main__":
    main()
