# Running the cross-machine and page cache experiments

Both reuse the main harness. No changes to `probe.py` or to the models.

## Replication across machines

The same benchmark runs on each machine. The `--env` flag keeps records
separate, so nothing overwrites anything.

On the first machine, from the repository root:

```bash
pip install -r requirements.txt
python scripts/capture_environment.py --env A --note "describe the runtime"
python scripts/prepare.py
python scripts/run_bench.py
python scripts/run_batch_sweep.py
```

On the second machine, the same sequence with the label changed:

```bash
python scripts/capture_environment.py --env B --note "describe the runtime"
python scripts/prepare.py
python scripts/run_bench.py       --env B
python scripts/run_batch_sweep.py --env B
```

Records for a labelled run land in `results/env_<label>/`. The first machine in
this package writes to the flat `results/` paths and carries the label `A` in
every row.

Run `capture_environment.py` before the benchmark on every machine. Hosted
runtimes are reassigned between sessions, so a machine's processor, core count
and memory cannot be recovered once the session ends.

Copy every `results/env_*/` folder back into one checkout, then:

```bash
python scripts/analyze.py
python scripts/crossover_analysis.py
python scripts/analyze_extensions.py
```

Time: about 40 minutes for `run_bench.py` and 40 for the sweep, per machine,
unattended and resumable.

## Page cache eviction

This measures how much a warm page cache flatters the main results. Four
eviction levels, interleaved so that machine drift cannot masquerade as an
effect:

| level | what it evicts | privileges |
|---|---|---|
| `warm` | nothing; the control arm | none |
| `artifact` | the model file only | none |
| `libs` | the model file plus the installed package trees | none |
| `full` | the entire system page cache | root |

Without root, on any machine including a hosted notebook:

```bash
python scripts/run_coldcache.py --env B --levels warm,artifact,libs
```

With root, on a machine you control:

```bash
sudo -E python scripts/run_coldcache.py --env local --levels warm,artifact,libs,full
```

`sudo -E` preserves the environment so the same Python is used. Time: roughly
25 minutes for the three no-root levels over the native and ONNX artifacts,
and about 10 minutes more for the `full` level.

To cover all four serialization formats rather than two, add
`--formats pickle,joblib,native,onnx`. That doubles the runtime.

The no-root levels rest on `posix_fadvise` with `POSIX_FADV_DONTNEED`, which is
advisory. The kernel may keep a page resident, and pages shared with another
process are not dropped at all. A null result from these levels is therefore
inconclusive rather than evidence of no effect. The `full` level, which writes
to `/proc/sys/vm/drop_caches`, is the one that settles the question, and it
needs a machine where that file is writable.

## What the reports contain

`scripts/analyze_extensions.py` prints both parts and writes
`results/extensions_report.txt`, plus `figures/fig7_environments.png` and
`figures/fig8_coldcache.png`.

The questions each part answers:

- **Replication**: does import dominance hold on other hardware, and do the
  crossover batch sizes move between machines?
- **Page cache**: how much of the ONNX cold start advantage survives when the
  artifact and the library files are evicted?
