"""Solver registry: ``{name: open_process}`` plus the named arms (presets) menu.

Contract (artifact documentation): one-shot ``solve(instance, params, budget, seed) ->
Trajectory`` and its resumable form ``open_process(instance, params, seed)``
-> process with ``advance(budget)`` (cumulative hard cap, see brdeh.py).
The run-driver drives processes. One adapter module per solver, no class
hierarchy.
"""

from . import brdeh

SOLVERS = {brdeh.SOLVER_NAME: brdeh.solve}
PROCESSES = {brdeh.SOLVER_NAME: brdeh.open_process}

# Arm = named preset = (solver, params). Policies choose among arm names.
ARMS = {name: (brdeh.SOLVER_NAME, params) for name, params in brdeh.PRESETS.items()}

# Submodule path of each solver, for provenance (commit + dirty flag).
SOLVER_PATHS = {brdeh.SOLVER_NAME: "solvers/br-deh"}

# Runtime facts of each solver (e.g. Julia version), for provenance.
ENVIRONMENT = {brdeh.SOLVER_NAME: brdeh.environment}
