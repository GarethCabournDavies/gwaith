# Residual event statistics

`hwilas_check_search_residuals` reports the highest candidate statistic outside
±`--signal-exclusion` seconds of every optimised event's `tc` in
`--parameter-file`. The default is 86400 seconds (one day on either side);
boundary events are excluded. Candidate times are the mean of the first two
requested channel times. An empty remaining catalogue gives a JSON `null`
maximum.

`--output-statistic-plot residual_statistics.png` plots cumulative event rates
N(statistic ≥ threshold) / searched seconds outside signal windows, in Hz on a
logarithmic y-axis, with no histogram binning or statistic-40 cut. The
existing unfiltered threshold counts remain available in the JSON report.
Both basic search examples request this plot; set `RESIDUAL_SIGNAL_EXCLUSION`
to change their exclusion window in seconds.

Noise-background estimates and uncertainty bands are deferred; time slides may
provide a background estimate in future.

The rate denominator unions candidate `analysis_intervals` (so template
partitions do not multiply the exposure), then removes the union of fitted
signal windows, clipped to those intervals. Without interval metadata it uses
the data span minus `--end-exclusion` and logs this fallback. Counts, rates in
Hz and exposure seconds are all saved in the JSON report.

The Mojito residual search runs with `RESIDUAL_STATISTIC_THRESHOLD=4` by default;
the earlier search rounds keep `NETWORK_STATISTIC_THRESHOLD=6`. The residual
search commands are enabled, so rerunning the workflow produces fresh residual
triggers. Changing the threshold requires rerunning that search, not just the
residual checker. The rate describes selected clustered candidates, not a
noise false-alarm rate.
