# Baseline experiment protocol

The baseline sweep uses Stim's `surface_code:rotated_memory_z` circuit with
`rounds = distance`, `after_clifford_depolarization = p`,
`after_reset_flip_probability = p`, `before_measure_flip_probability = p`, and
`before_round_data_depolarization = p`. The grid is
`p = {0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.03}` for `d = {3, 5, 7}`.

The checked-in script defaults to 10,000 shots per point and base seed
20261028. Each point gets seed `base_seed + 100000*d + rate_index`, and the seed,
shot count, logical-error count, Wilson 95% interval, and decode timing are
written to CSV. Timing is the median of three full-batch decodes after one
256-shot warm-up; it excludes circuit creation, sampling, and DEM compilation.
It is therefore a decoder-only latency baseline. Re-run on the same machine
and environment for timing comparisons.

Run with:

```powershell
python -m src.experiments.baseline_curve --output data/baseline_curve.csv --shots 10000 --seed 20261028
python -m src.experiments.plot_baseline_curve --input data/baseline_curve.csv --output data/baseline_curve.png
```

The 10,000-shot sweep is an initial reference, not a precision threshold
estimate: zero-error points have only an upper confidence bound, and finer
threshold estimation needs substantially more shots near the crossing.

# Hybrid predecoder contract

`SyndromePredecoder.predecode` receives boolean detector events shaped
`(shots, detectors)` and returns boolean residual events of the same shape plus
`observable_corrections` shaped `(shots, observables)`. The hybrid decoder
passes residual events to PyMatching and XOR-composes its observable prediction
with the predecoder correction. A defect may only be removed when its implied
correction is represented in that output; otherwise the logical error rate can
be biased. `IdentityPredecoder` is the no-op control for integration and timing,
not a neural result.
