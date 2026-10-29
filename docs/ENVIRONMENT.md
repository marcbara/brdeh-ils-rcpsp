# Environment

The tested interpreter is Python 3.12.14 with `juliacall`, Julia 1.11.9, and the pinned Julia project/manifest at repository root. Julia packages are pinned in `Manifest.toml`; no global install is required. Set `JULIA_DEPOT_PATH` to a depot holding that exact environment, then run commands from this repository root. When `juliacall` is configured with an explicit executable, set both `PYTHON_JULIACALL_EXE=$(command -v julia)` and `PYTHON_JULIACALL_PROJECT=$PWD`; this prevents it from trying to mutate the Python environment's project.

The current verification used an existing local depot only to exercise the artifact. That depot is not a runtime path or package dependency. A clean-machine install has not yet been performed, so it is explicitly unverified.

Single-thread settings are recommended for comparable timing: `JULIA_NUM_THREADS=1`, `JULIA_NUM_GC_THREADS=1`, `PYTHON_JULIACALL_THREADS=1`. Smoke records CPU and monotonic wall time around each solve. Julia JIT warm-up occurs before timed V0 work; parsing and JIT are not counted as solver work by the historical driver.
