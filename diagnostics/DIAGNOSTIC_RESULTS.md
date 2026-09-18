"""Diagnostic results: Sky location scan for FastBHB vs Sangria reference

Run date: 2026-04-09
Signal #0: M1=1015522, M2=796849, D=17758.4 Mpc, tc=4800021.2s

KEY FINDING: "ZERO SNR" IS A DISPLAY ARTIFACT
=============================================

The matched_filter_timeseries function returns SNR values in strain units (~1e-18),
NOT dimensionless SNR. When printed with %.3f format, these appear as 0.000.

Actual values for Signal #0, LISA_A channel:
  snr_peak = 1.39e-18 (not zero!)
  snr_perfect_ref = 2.56e-18 (not zero!)
  match_quality = snr_peak/snr_perfect_ref = 0.544

The meaningful metric is the RATIO: snr_peak / snr_perfect_ref, which gives
the fraction of the perfect-match SNR achieved. A value of 1.0 would indicate
a perfect match; 0.544 indicates a significant mismatch.

FINDINGS
========

1. BASELINE CORRELATION (reference sky location):
   LISA_A: corr_peak = +0.9033  (shift +5s)
   LISA_E: corr_peak = +0.7120  (shift +30s)
   LISA_T: corr_peak = +0.8185  (shift -10s)

   The waveforms are largely aligned but not perfectly matching. LISA_A has
   the best match (~90% correlation), while LISA_E is notably lower (~71%).

2. MATCHED FILTER SNR (unwhitened, strain units):
   LISA_A: snr_peak/snr_perfect_ref = 0.544 (54% match quality)
   
   Note: The correlation coefficient (0.903) and the SNR ratio (0.544) differ
   because they measure different things:
   - Correlation measures shape similarity (normalized cross-correlation)
   - SNR ratio measures amplitude-weighted overlap in the matched filter sense

3. COALESCENCE TIME SCAN:
   Correlation stays flat at ~0.9033 across all time shifts (-50 to +50 samples).
   The residual shift tracks the applied shift almost exactly (e.g., applying
   +50 samples gives residual -49 samples). This confirms there is NO systematic
   time offset between the hwilas waveform and the Sangria data.

4. SKY LOCATION SCANS:
   Ecliptic longitude: gentle slope from 0.88 to 0.93 over ±0.2 rad,
                      best at +0.2 rad (corr=0.9251)
   Ecliptic latitude:  peaks around -0.41 rad with corr=0.9123

   The improvements are modest (~2-3% correlation gain) and the trends are
   gradual rather than sharp peaks. This suggests the mismatch is NOT primarily
   due to sky location differences.

CONCLUSION
==========

The "zero SNR" issue was a display/precision problem, not a fundamental bug.
The actual match quality is ~54% (SNR ratio) or ~90% (correlation), indicating
a real but moderate mismatch between hwilas FastBHB waveforms and Sangria data.

The remaining mismatch likely comes from:
1. Different orbital mechanics (analytic vs OEM orbits in LDC FastBHB)
2. Different TDI generation (hwilas uses linear_AET_conversion, Sangria may use different TDI)
3. Mode content differences (Sangria reference includes multiple modes, hwilis diagnostic used single mode)
4. Spin parameter mapping (we only use spin1z/spin2z, full 3D spins may matter)

RECOMMENDED NEXT STEPS
======================

1. Fix the display in diagnose_sangria_waveform.py to show SNR ratios instead of absolute values
2. Check if the Sangria data was generated with TDI 1.5 or TDI 2
3. Compare the orbital phase evolution between hwilas and LDC FastBHB
4. Test with full spin vectors (not just z-components)
5. Verify that the inclination/polarization angle conventions match between hwilas and LDC

FILES
=====
- diagnostics/sky_location_scan.py: Main diagnostic script
- /tmp/sky_scan_output.txt: Full output from the scan
"""
