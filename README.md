# Measuring Cold Start Cost of Serialization Formats in Gradient Boosted Tree Deployment

Replication package for the paper of the same name.

The study measures what a gradient boosted tree model costs to **deploy** rather
than what it costs to **run**. Steady-state throughput benchmarks assume a warm
process where the library is imported and the model already sits in memory.
Serverless functions, autoscaled containers, scheduled batch jobs and command
line tools all start from a fresh process instead. This package contains the
measurement harness, the raw measurement records and the analysis code needed to
reproduce every table and figure in the paper.

## Headline findings

| | |
|---|---|
| Measurements, main experiment and batch sweep, two machines | 5,040 |
| Library import as a share of time to first prediction | 88.2% to 94.1% |
| Artifact deserialization as a share | about 0.5% |
| Artifact size range across the grid | 326x |
| Cold start range for in-library formats | 2.31x |
| ONNX reduction in time to first prediction | 7.35x on A, 8.09x on B |
| Import share shift between the two machines | at most 1.1 points |
| Crossover batch size, CatBoost | 8 to 16 |
| Crossover batch size, XGBoost | 32 to 512 on A, 64 to 2048 on B |
| Crossover batch size, LightGBM | none up to 2048 on either machine |

## Reproducing the paper without re-running anything

The measurement records are included, so the published numbers can be checked in
under a minute:

```bash
pip install -r requirements.txt
python scripts/analyze.py              # Tables 3, 4, 5, 7, 8 and Figures 1 to 4
python scripts/crossover_analysis.py   # Table 6 and the crossover figure
python scripts/analyze_extensions.py   # cross-machine replication, page cache
python scripts/make_method_figures.py  # the two method diagrams
```

Tables land in `results/` as one CSV per numbered table, figures in `figures/`,
and a plain text summary of the headline numbers in `results/analysis_report.txt`.

## Reproducing the measurements from scratch

Requires Linux and Python 3.10 or later. A free hosted notebook CPU runtime is
enough. No GPU, no elevated privileges.

```bash
python scripts/prepare.py --quick      # 1 minute, checks the toolchain
python scripts/run_bench.py --quick    # 3 minutes, writes raw_smoketest.jsonl

python scripts/capture_environment.py --env A --note "describe the runtime"
python scripts/prepare.py              # trains 24 models, exports 96 artifacts
python scripts/run_bench.py            # 1,440 measurements, about 40 minutes
python scripts/run_batch_sweep.py      # 1,080 measurements, about 40 minutes
python scripts/analyze.py
python scripts/crossover_analysis.py
```

To add a second machine, run the same sequence there with `--env B` on
`capture_environment.py`, `run_bench.py` and `run_batch_sweep.py`. Records then
land in `results/env_B/` and the analysis scripts pick them up on their own.

`prepare.py` downloads Covertype on first use. Both measurement scripts append to
disk after every measurement and skip completed cells on restart, so an
interrupted run can be resumed with the same command.

For Google Colab, `notebooks/coldstart_benchmark.ipynb` runs the same pipeline
and persists results to Google Drive.

## How the measurement works

Every measurement runs in its own operating system process. This is the design
decision the whole study rests on: a warm interpreter has already paid the
import cost, so measuring cold start inside a loop would report it as near zero.
`scripts/probe.py` therefore takes exactly one measurement per invocation and
prints a single JSON record. It is never imported by the orchestrator.

Time to first prediction is decomposed into five stages, each timed separately:

1. Bare interpreter start-up, established once by launching a process that does nothing
2. `import numpy`, timed alone because every backend pays it
3. Import of the model library or of ONNX Runtime
4. Deserialization of the artifact
5. The first `predict` call, which includes thread-pool creation

Steady-state latency is then measured with 100 further predictions in the same
process. The two regimes are recorded separately and never pooled.

Controls: thread counts are pinned before NumPy loads, so no format can benefit
from grabbing more cores; the input batch is identical across formats within a
cell; training is seeded; five repeats per cell with medians reported.

Shares of time to first prediction are computed from stage medians rather than
from the median of per-record ratios, so the five stages always add to the whole.
`analyze.py`, `analyze_extensions.py` and the paper all use that convention.

## Experimental design

| Factor | Levels |
|---|---|
| Library | LightGBM, XGBoost, CatBoost |
| Format | pickle, joblib, library native, ONNX |
| Dataset | Breast Cancer, California Housing, Covertype |
| Model size | 100, 500, 2000 trees (100, 500 for Covertype) |
| Batch size, main run | 1, 32, 1024 |
| Batch size, sweep | 1 to 2048 in powers of two |
| Machine | two independent hosted runtimes |
| Repeats | 5 independent process launches per cell |

## Repository layout

```
scripts/
  prepare.py              train models, export all four formats, write manifest
  probe.py                one cold-start measurement in one fresh process
  run_bench.py            main benchmark orchestrator, resumable
  run_batch_sweep.py      fine-grained batch sweep for the crossover estimate
  run_coldcache.py        page cache eviction arms, posix_fadvise based
  capture_environment.py  record processor, memory and package versions
  analyze.py              Tables 3, 4, 5, 7, 8 and Figures 1 to 4
  crossover_analysis.py   Table 6 and the crossover figure, bootstrap intervals
  analyze_extensions.py   cross-machine replication and the page cache arms
  make_method_figures.py  the two method diagrams, drawn from coordinates
notebooks/
  coldstart_benchmark.ipynb   Colab pipeline with Google Drive persistence
results/
  manifest.csv            one row per artifact: size, export time, status
  raw_main.jsonl          1,440 measurements, main factorial run, machine A
  raw_batchsweep.jsonl    1,080 measurements, batch sweep, machine A
  raw_smoketest.jsonl     48 measurements from the smoke-test configuration
  env_A/environment.json  what was recorded about the first machine
  env_B/environment.json  processor, cores, memory and versions, second machine
  env_B/raw_main.jsonl    1,440 measurements, main factorial run, machine B
  env_B/raw_batchsweep.jsonl  1,080 measurements, batch sweep, machine B
  env_B/raw_coldcache.jsonl   720 measurements, page cache eviction arms
  summary.csv             median of every metric per cell, both machines
  table3_composition.csv  stage breakdown of time to first prediction
  table4_variance.csv     variance explained by each factor alone
  table5_equivalence.csv  paired equivalence tests among in-library formats
  table6_crossover.csv    crossover batch size with bootstrap intervals
  table7_environments.csv the two measurement environments side by side
  table8_replication.csv  headline numbers on both machines
  crossover_curves.csv    ratio against batch size, both machines
  crossover_bootstrap.csv crossover estimate per cell, both machines
  equivalence.csv         ONNX against native prediction agreement
  analysis_report.txt     headline numbers in plain text
  extensions_report.txt   replication and page cache results in plain text
  versions.json           package versions, pointing at the per-machine files
figures/
paper/
```

Each experimental design writes to its own record file, and measurements taken
under different designs are never pooled. Every row carries an `env_label` and an
`experiment` field, so the records stay self-describing once concatenated.

## Environment

Measurements were taken on two independent hosted Linux notebook runtimes on
x86-64. `results/env_A/environment.json` and `results/env_B/environment.json`
record what is known about each.

| | Machine A | Machine B |
|---|---|---|
| Processor | not recorded | Intel Xeon at 2.20 GHz |
| Logical cores | not recorded | 2 |
| Memory | not recorded | 12.67 GB |
| Kernel | 6.6.122+ | 6.6.122+ |
| C library | glibc 2.35 | glibc 2.39 |
| Python | 3.13.15 | 3.13.15 |
| LightGBM, XGBoost, CatBoost | 4.6.0, 3.4.1, 1.2.10 | 4.6.0, 3.4.1, 1.2.10 |
| ONNX Runtime | 1.29.0 | 1.30.0 |

Absolute timings depend on hardware, operating system and library versions. The
relative comparisons are made within one machine and then checked against the
other. Anyone reproducing this on different hardware should expect different
absolute numbers.

## Known limitations

These are stated in the paper and repeated here so that anyone reusing the
harness knows what it does and does not measure.

- **Warm page cache.** The headline measurements are process-level cold starts
  with the artifact file already in memory. `run_coldcache.py` evicts the
  artifact and the library files with `posix_fadvise`, which needs no elevated
  privileges, and finds no significant difference on machine B: every bootstrap
  interval contains one. That is an inconclusive result rather than evidence of
  no effect, because `posix_fadvise` is advisory and a genuine first load from a
  cold disk could still differ.
- **Two hosted runtimes, not a hardware matrix.** Hardware details for machine A
  were not captured at the time of measurement and could not be recovered, so
  this is a replication across two runtimes rather than a controlled hardware
  comparison.
- **ONNX Runtime differs between the machines**, 1.29.0 on A and 1.30.0 on B,
  as does the C library version. Any crossover estimate involving ONNX carries
  that confound. The import decomposition does not, because it does not involve
  ONNX at all.
- **Threads pinned to one** for comparability. Multi-threaded serving shifts the
  steady-state numbers, though not the import decomposition.
- **The ONNX path changes both the artifact and the runtime.** Prediction is
  executed by ONNX Runtime rather than by the training library. The stage
  decomposition shows where the gain comes from, but the design cannot separate
  the two effects entirely.
- **Model saturation.** LightGBM produces identical artifacts at 500 and 2000
  trees on Breast Cancer, because 398 training rows cannot support further splits
  under the default minimum leaf size. That cell is excluded from size-related
  claims.
- **Memory.** `probe.py` records peak resident set size from `/proc/self/status`.
  Some container runtimes report a process-independent figure through the
  `getrusage` interface, which is why this reading is taken from `/proc`. Memory
  is not reported in the paper.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
