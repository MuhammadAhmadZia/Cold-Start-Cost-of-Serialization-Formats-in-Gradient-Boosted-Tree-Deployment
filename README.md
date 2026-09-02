# Cold Start Cost of Serialization Formats in Gradient Boosted Tree Deployment

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
| Library import as a share of time to first prediction | 88.1% to 93.5% |
| Artifact deserialization as a share | about 0.5% |
| Artifact size range across the grid | 326x |
| Cold start range for in-library formats | 2.31x |
| ONNX reduction in time to first prediction | 7.35x |
| Crossover batch size, CatBoost | 8 to 16 |
| Crossover batch size, XGBoost | 32 to 512 |
| Crossover batch size, LightGBM | none up to 2048 |

## Reproducing the paper without re-running anything

The measurement records are included, so the published numbers can be checked in
under a minute:

```bash
pip install -r requirements.txt
python scripts/analyze.py              # Tables 3, 4, 5 and Figures 3, 4, 5
python scripts/crossover_analysis.py   # Table 6 and Figure 6
python scripts/make_method_figures.py  # Figures 1 and 2
```

Results land in `results/` and figures in `figures/`.

## Reproducing the measurements from scratch

Requires Linux and Python 3.10 or later. A free hosted notebook CPU runtime is
enough. No GPU, no elevated privileges.

```bash
python scripts/prepare.py --quick      # 1 minute, checks the toolchain
python scripts/run_bench.py --quick    # 3 minutes, writes raw_smoketest.jsonl

python scripts/prepare.py              # trains 24 models, exports 96 artifacts
python scripts/run_bench.py            # 1,440 measurements, about 40 minutes
python scripts/run_batch_sweep.py      # 1,080 measurements, about 40 minutes
python scripts/analyze.py
python scripts/crossover_analysis.py
```

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

## Experimental design

| Factor | Levels |
|---|---|
| Library | LightGBM, XGBoost, CatBoost |
| Format | pickle, joblib, library native, ONNX |
| Dataset | Breast Cancer, California Housing, Covertype |
| Model size | 100, 500, 2000 trees (100, 500 for Covertype) |
| Batch size, main run | 1, 32, 1024 |
| Batch size, sweep | 1 to 2048 in powers of two |
| Repeats | 5 independent process launches per cell |

## Repository layout

```
scripts/
  prepare.py              train models, export all four formats, write manifest
  probe.py                one cold-start measurement in one fresh process
  run_bench.py            main benchmark orchestrator, resumable
  run_batch_sweep.py      fine-grained batch sweep for the crossover estimate
  analyze.py              Tables 3, 4, 5 and Figures 3, 4, 5
  crossover_analysis.py   Table 6 and Figure 6, bootstrap intervals
  make_method_figures.py  Figures 1 and 2, drawn from coordinates
notebooks/
  coldstart_benchmark.ipynb   Colab pipeline with Google Drive persistence
results/
  manifest.csv            one row per artifact: size, export time, status
  raw_main.jsonl          1,440 measurements, main factorial run
  raw_batchsweep.jsonl    1,080 measurements, batch sweep
  raw_smoketest.jsonl     48 measurements from the smoke-test configuration
  equivalence.csv         ONNX against native prediction agreement
  versions.json           exact package versions used
figures/
paper/
```

Each experimental design writes to its own record file. Measurements taken under
different designs are never pooled, and `analyze.py` reads only `raw_main.jsonl`.

## Environment

Measurements in the paper were taken on a single hosted Linux notebook runtime on
x86-64 with the versions recorded in `results/versions.json`: Python 3.13.15,
LightGBM 4.6.0, XGBoost 3.4.1, CatBoost 1.2.10, ONNX Runtime 1.29.0,
scikit-learn 1.6.1 and NumPy 2.1.3.

Absolute timings depend on hardware, operating system and library versions. The
relative comparisons are made within one environment. Anyone reproducing this on
different hardware should expect different absolute numbers.

## Known limitations

These are stated in the paper and repeated here so that anyone reusing the
harness knows what it does and does not measure.

- **Warm page cache.** Dropping the operating system page cache needs privileges
  the environment does not grant, so these are process-level cold starts with the
  artifact file already in memory. A genuine first load from disk would differ.
- **Single machine.** One hardware and software configuration.
- **Threads pinned to one** for comparability. Multi-threaded serving shifts the
  steady-state numbers, though not the import decomposition.
- **The ONNX path changes both the artifact and the runtime.** Prediction is
  executed by ONNX Runtime rather than by the training library. This design
  cannot separate the two effects.
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
