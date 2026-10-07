"""
capture_environment.py - record everything about the machine that could change
a timing result.

Run this once on every platform before the benchmark. The output is written to
results/env_<label>/environment.json and is what lets the paper say precisely
what "a single hosted notebook environment" was, and how the environments
differ from each other.

Run:  python scripts/capture_environment.py --env colab
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def read_first(path, pattern):
    try:
        with open(path) as f:
            for line in f:
                m = re.match(pattern, line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return None


def cpu_info():
    info = {
        "model": read_first("/proc/cpuinfo", r"model name\s*:\s*(.+)"),
        "vendor": read_first("/proc/cpuinfo", r"vendor_id\s*:\s*(.+)"),
        "flags_avx512": False,
        "logical_cores": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        info["flags_avx512"] = "avx512f" in txt
        info["physical_cores"] = len(set(re.findall(r"core id\s*:\s*(\d+)", txt)))
    except OSError:
        pass
    try:
        info["affinity_cores"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["affinity_cores"] = None
    return info


def mem_info():
    kb = read_first("/proc/meminfo", r"MemTotal:\s*(\d+) kB")
    out = {"total_gb": round(int(kb) / 1048576, 2) if kb else None}
    # Container memory ceiling, which is usually lower than the host total.
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if os.path.exists(p):
            try:
                v = open(p).read().strip()
                if v.isdigit() and int(v) < (1 << 50):
                    out["cgroup_limit_gb"] = round(int(v) / (1 << 30), 2)
            except OSError:
                pass
            break
    return out


def storage_info():
    art = os.path.join(ROOT, "artifacts")
    target = art if os.path.isdir(art) else ROOT
    out = {"artifacts_path": target}
    try:
        st = shutil.disk_usage(target)
        out["free_gb"] = round(st.free / (1 << 30), 2)
    except OSError:
        pass
    try:
        r = subprocess.run(["stat", "-f", "-c", "%T", target],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out["filesystem"] = r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def package_versions():
    out = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost",
                "catboost", "onnxruntime", "onnxmltools", "joblib",
                "statsmodels"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = None
    return out


def eviction_capability():
    """Which cache-eviction methods this machine allows."""
    out = {"euid": os.geteuid(), "posix_fadvise": hasattr(os, "posix_fadvise")}
    p = "/proc/sys/vm/drop_caches"
    out["drop_caches_present"] = os.path.exists(p)
    out["drop_caches_writable"] = os.access(p, os.W_OK)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True,
                    help="short label for this platform, e.g. colab, kaggle, local")
    ap.add_argument("--note", default="",
                    help="free-text note, e.g. 'Colab CPU runtime, free tier'")
    args = ap.parse_args()

    out_dir = os.path.join(RES, f"env_{args.env}")
    os.makedirs(out_dir, exist_ok=True)

    env = {
        "env_label": args.env,
        "note": args.note,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "cpu": cpu_info(),
        "memory": mem_info(),
        "storage": storage_info(),
        "packages": package_versions(),
        "eviction": eviction_capability(),
    }

    path = os.path.join(out_dir, "environment.json")
    with open(path, "w") as f:
        json.dump(env, f, indent=2)

    print(f"environment recorded -> {path}\n")
    print(f"  label        {env['env_label']}")
    print(f"  cpu          {env['cpu']['model']}")
    print(f"  cores        {env['cpu']['logical_cores']} logical, "
          f"{env['cpu'].get('affinity_cores')} available to this process")
    print(f"  memory       {env['memory']['total_gb']} GB"
          + (f" (cgroup limit {env['memory']['cgroup_limit_gb']} GB)"
             if env['memory'].get('cgroup_limit_gb') else ""))
    print(f"  filesystem   {env['storage'].get('filesystem')}")
    print(f"  python       {env['python']}")
    print(f"  eviction     fadvise={env['eviction']['posix_fadvise']}, "
          f"drop_caches={env['eviction']['drop_caches_writable']}")


if __name__ == "__main__":
    main()
