#!/usr/bin/env python3
"""Diagnostic: scan tc offsets for the PhenomX TDI waveform against Sangria.

For signal #0, regenerate the PhenomXPy + TDISpline waveform with a range of
tc offsets (tc + delta) and measure the matched-filter SNR ratio against the
noiseless injection stream. A perfect regeneration peaks at delta = 0; a
systematic offset shows up as a peak elsewhere.

Also tests sky-position hypotheses:
  * baseline: catalog eclipticlongitude/eclipticlatitude as-is
  * swapped:  (eclipticlatitude, eclipticlongitude)
  * shifted:  longitude + pi/2
"""
import numpy as np
import h5py

from hwilas.data_processing.tdi import linear_AET_conversion
from hwilas.waveforms.phenomx_tdi import (
    DEFAULT_HM_MODE_ARRAY_TD,
    get_td_waveform_from_phenomx,
)
from hwilas.filter import matched_filter_timeseries, sigmasq_frequencyseries
from gwaith.types import TimeSeries

DATA_FILE = "examples/datasets/LDC2_sangria_hm_training.hdf"


def _scalar(v):
    return float(np.asarray(v).ravel()[0])


def load_signal(filename, idx=0):
    with h5py.File(filename, "r") as f:
        attrs = dict(f["obs/tdi"].attrs)
        delta_t = float(attrs["dt"])
        epoch = float(attrs.get("t0", 0.0))
        ds = f["sky/mbhb/tdi"]
        X = np.asarray(ds["X"], dtype=float).ravel()
        Y = np.asarray(ds["Y"], dtype=float).ravel()
        Z = np.asarray(ds["Z"], dtype=float).ravel()
        rec = f["sky/mbhb/cat"][idx]

    aet = linear_AET_conversion(X, Y, Z)
    spin1_mag = _scalar(rec["Spin1"])
    spin2_mag = _scalar(rec["Spin2"])
    sig = {
        "mass1": _scalar(rec["Mass1"]),
        "mass2": _scalar(rec["Mass2"]),
        "spin1z": spin1_mag * np.cos(_scalar(rec["PolarAngleOfSpin1"])),
        "spin2z": spin2_mag * np.cos(_scalar(rec["PolarAngleOfSpin2"])),
        "distance": _scalar(rec["Distance"]),
        "eclipticlongitude": _scalar(rec["EclipticLongitude"]),
        "eclipticlatitude": _scalar(rec["EclipticLatitude"]),
        "inclination": _scalar(rec["InitialPolarAngleL"]),
        "polarization": _scalar(rec["InitialAzimuthalAngleL"]),
        "phi_ref": _scalar(rec["PhaseAtCoalescence"]),
        "tc": _scalar(rec["CoalescenceTime"]),
    }
    return aet, sig, delta_t, epoch


def generate(aet, sig, delta_t, tc_offset=0.0, sky_variant="baseline",
             window_before_days=1.0, window_after_days=0.1):
    tc = sig["tc"] + tc_offset
    n_samples = len(aet["LISA_A"])
    times = 0.0 + delta_t * np.arange(n_samples)
    lo = max(times[0], tc - window_before_days * 86400)
    hi = min(times[-1], tc + window_after_days * 86400)
    n = int(round((hi - lo) / delta_t)) + 1
    gen_times = np.linspace(lo, hi, n)

    if sky_variant == "baseline":
        lon, lat = sig["eclipticlongitude"], sig["eclipticlatitude"]
    elif sky_variant == "swapped":
        lon, lat = sig["eclipticlatitude"], sig["eclipticlongitude"]
    elif sky_variant == "lon_plus_pi2":
        lon, lat = sig["eclipticlongitude"] + np.pi / 2, sig["eclipticlatitude"]
    else:
        raise ValueError(sky_variant)

    xyz = get_td_waveform_from_phenomx(
        detector_times=gen_times,
        tc=tc,
        mass1=sig["mass1"],
        mass2=sig["mass2"],
        spin1z=sig["spin1z"],
        spin2z=sig["spin2z"],
        distance=sig["distance"],
        inclination=sig["inclination"],
        phi_ref=sig["phi_ref"],
        eclipticlongitude=lon,
        eclipticlatitude=lat,
        polarization=sig["polarization"],
        mode_array=DEFAULT_HM_MODE_ARRAY_TD,
        orbit="analytic",
    )
    X = np.asarray(xyz["LISA_X"].data, dtype=float)
    Y = np.asarray(xyz["LISA_Y"].data, dtype=float)
    Z = np.asarray(xyz["LISA_Z"].data, dtype=float)
    model_aet = linear_AET_conversion(X, Y, Z)

    full = {}
    for channel, arr in model_aet.items():
        out = np.zeros(n_samples)
        lo_idx = int(np.searchsorted(times, gen_times[0], side="left"))
        hi_idx = int(np.searchsorted(times, gen_times[-1], side="right"))
        n_copy = min(hi_idx - lo_idx, len(arr))
        if n_copy > 0:
            out[lo_idx:lo_idx + n_copy] = arr[:n_copy]
        full[channel] = out
    return full, gen_times


def snr_ratio(channel_data, channel_model, delta_t, epoch=0.0):
    data_ts = TimeSeries(channel_data, epoch=epoch, delta_t=delta_t)
    model_ts = TimeSeries(channel_model, epoch=epoch, delta_t=delta_t)
    snr = matched_filter_timeseries(model_ts, data_ts, n_samples=len(data_ts))
    snr_peak = float(np.max(np.abs(snr.data)))
    sigma_sq = sigmasq_frequencyseries(model_ts.to_frequencyseries())
    ref = float(np.sqrt(max(sigma_sq, 0.0)))
    return snr_peak / ref if ref > 0 else 0.0




if __name__ == "__main__":
    aet, sig, delta_t, epoch = load_signal(DATA_FILE)
    print(f"Signal #0: M1={sig['mass1']:.0f} M2={sig['mass2']:.0f} "
          f"D={sig['distance']:.0f} tc={sig['tc']:.1f}s")
    print(f"  lon={sig['eclipticlongitude']:.4f} lat={sig['eclipticlatitude']:.4f} "
          f"incl={sig['inclination']:.4f} psi={sig['polarization']:.4f} "
          f"phi_ref={sig['phi_ref']:.4f}")

    ref_snr = snr_ratio(aet["LISA_A"], aet["LISA_A"], delta_t, epoch)
    print(f"\nReference self-SNR ratio (LISA_A): {ref_snr:.4f}")

    deltas = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    print(f"\n{'tc_offset':>10} | " + " | ".join(
        f"{ch:>8}" for ch in ("LISA_A", "LISA_E", "LISA_T")))
    print("-" * 60)
    for d in deltas:
        model, _ = generate(aet, sig, delta_t, tc_offset=d, sky_variant="baseline")
        row = [snr_ratio(aet[ch], model[ch], delta_t, epoch)
               for ch in ("LISA_A", "LISA_E", "LISA_T")]
        print(f"{d:>10} | " + " | ".join(f"{r:>8.4f}" for r in row))

    print("\nSky-variant comparison at tc_offset=0:")
    for variant in ("swapped", "lon_plus_pi2"):
        model, _ = generate(aet, sig, delta_t, tc_offset=0, sky_variant=variant)
        row = [snr_ratio(aet[ch], model[ch], delta_t, epoch)
               for ch in ("LISA_A", "LISA_E", "LISA_T")]
        print(f"  {variant:>12}: " + " | ".join(
            f"{ch}={r:.4f}" for ch, r in zip(("LISA_A", "LISA_E", "LISA_T"), row)))

    print("\nphi_ref scan at tc_offset=0:")
    for dphi in [0, np.pi / 2, -np.pi / 2, np.pi]:
        sig2 = dict(sig)
        sig2["phi_ref"] = sig["phi_ref"] + dphi
        model, _ = generate(aet, sig2, delta_t, tc_offset=0, sky_variant="baseline")
        row = [snr_ratio(aet[ch], model[ch], delta_t, epoch)
               for ch in ("LISA_A", "LISA_E", "LISA_T")]
        print(f"  dphi={dphi:+.3f}: " + " | ".join(
            f"{ch}={r:.4f}" for ch, r in zip(("LISA_A", "LISA_E", "LISA_T"), row)))

    # Test: pass polarization_angle to PhenomXPy (currently dropped!)
    print("\n\n=== POLARIZATION_ANGLE TEST ===")
    print("Testing whether passing 'polarization' as polarization_angle to PhenomXPy fixes the SNR.")
    from hwilas.waveforms.phenomx_tdi import build_phenomx_parameters
    import astropy.units as u

    # Monkey-patch generate_tdi_waveform to include polarization_angle
    import hwilas.waveforms.phenomx_tdi as px_mod
    orig_build = px_mod.build_phenomx_parameters

    def build_with_psi(*, m1_msun, m2_msun, chi1z, chi2z, distance_mpc,
                       inclination, phi_ref, delta_t, f_min_hz=1e-4,
                       f_ref_hz=None, mode_array=None, polarization_angle=0.0):
        if mode_array is None:
            mode_array = px_mod.DEFAULT_HM_MODE_ARRAY_TD
        if f_ref_hz is None:
            f_ref_hz = f_min_hz
        m1 = m1_msun * u.solMass
        m2 = m2_msun * u.solMass
        mass_ratio = m1 / m2
        from phenomxpy.gwsignal_wrapper import convert_params_from_gwsignal
        return convert_params_from_gwsignal({
            "eta": mass_ratio / (1.0 + mass_ratio) ** 2,
            "total_mass": m1 + m2,
            "mass_ratio": mass_ratio,
            "s1": [0.0 * u.dimensionless_unscaled] * 2 + [chi1z * u.dimensionless_unscaled],
            "s2": [0.0 * u.dimensionless_unscaled] * 2 + [chi2z * u.dimensionless_unscaled],
            "distance": distance_mpc * u.Mpc,
            "inclination": inclination * u.rad,
            "phi_ref": phi_ref * u.rad,
            "polarization_angle": polarization_angle * u.rad,
            "delta_t": delta_t * u.s,
            "condition": 0,
            "f_min": f_min_hz * u.Hz,
            "f_ref": f_ref_hz * u.Hz,
            "mode_array": mode_array,
        })

    # Patch generate_tdi_waveform to pass polarization_angle
    import inspect
    src = inspect.getsource(px_mod.generate_tdi_waveform)
    print(f"\n  Current generate_tdi_waveform does NOT pass polarization_angle to PhenomXPy.")
    print(f"  Catalog polarization (psi) = {sig['polarization']:.4f} rad")

    # Direct test: manually build with and without polarization_angle
    from phenomxpy import IMRPhenomTHM
    from scipy.interpolate import CubicSpline
    from tdispline import TDISpline, X2
    from hwilas.waveforms.phenomx_tdi import get_analytic_orbit_interpolant

    tc = sig["tc"]
    lo = max(times[0], tc - 1.0 * 86400)
    hi = min(times[-1], tc + 0.1 * 86400)
    gen_times = np.linspace(lo, hi, int(round((hi - lo) / delta_t)) + 1)

    orbits = get_analytic_orbit_interpolant(gen_times)
    tdi = TDISpline(orbits, combination=X2)
    tdi.set_extrinsic_parameters(sig["eclipticlongitude"], sig["eclipticlatitude"])

    from phenomxpy.utils import SpinWeightedSphericalHarmonic

    for label, psi_val in [("no_psi (current)", 0.0), ("with_psi", sig["polarization"])]:
        params = build_with_psi(
            m1_msun=sig["mass1"], m2_msun=sig["mass2"],
            chi1z=sig["spin1z"], chi2z=sig["spin2z"],
            distance_mpc=sig["distance"], inclination=sig["inclination"],
            phi_ref=sig["phi_ref"], delta_t=delta_t,
            polarization_angle=psi_val,
        )
        gen = IMRPhenomTHM(**params)
        wtimes = np.linspace(gen_times[0] - tc - 1000, gen_times[-1] - tc + 1000, 8192)
        hlms, _ = gen.compute_hlms(wtimes)
        mode_keys = list(hlms)
        harmonics = np.array([
            SpinWeightedSphericalHarmonic(
                gen.pWF.inclination, np.pi/2 - gen.pWF.phi_ref, *m
            ) for m in [tuple(map(int, k.replace('-', ' ').split())) if '-' in k else (int(k[0]), int(k[1:])) for k in mode_keys]
        ])
        # Fix: parse mode keys properly
        from hwilas.waveforms.phenomx_tdi import _mode_indices
        harmonics = np.array([
            SpinWeightedSphericalHarmonic(
                gen.pWF.inclination, np.pi/2 - gen.pWF.phi_ref, *_mode_indices(k)
            ) for k in mode_keys
        ])
        mw = np.column_stack([hlms[k] for k in mode_keys]) * harmonics
        stimes = wtimes + tc
        amp = CubicSpline(stimes, np.abs(mw), axis=0)
        phs = CubicSpline(stimes, np.unwrap(np.angle(mw), axis=0), axis=0)
        rtimes = np.linspace(gen_times[0], gen_times[-1], 4096)
        tdi_modes = tdi(rtimes, amp, phs, nmodes=len(mode_keys), new_t=gen_times)
        xyz = tdi_modes.sum(axis=-1)
        # XYZ to AET
        from hwilas.data_processing.tdi import linear_AET_conversion
        aet_model = linear_AET_conversion(xyz[0], xyz[1], xyz[2])

        row = [snr_ratio(aet[ch], np.asarray(aet_model[ch]), delta_t, epoch)
               for ch in ("LISA_A", "LISA_E", "LISA_T")]
        print(f"  {label:>20}: " + " | ".join(
            f"{ch}={r:.4f}" for ch, r in zip(("LISA_A", "LISA_E", "LISA_T"), row)))
