# Actual execution results

Run on November 25, 2026, using Python 3.14.7 on macOS ARM64.
Installed: Qiskit 2.6.2, SciPy 1.18.1, NumPy 2.5.5.
The project-local `.venv` was created and these dependencies were installed.
`python -m pip check` reported no broken requirements.
Exact installed dependencies are in `requirements-lock.txt`.

## Tests

Command: `python -m unittest -v`

All **5 tests passed** in 0.047 seconds. They cover every binary state for both
problem sizes, baseline costs and route validity, phase encoding, mixer
behavior, and a short optimization run.

## Four-city example

Command: `python route_qaoa.py`

9 qubits, 2 layers, 3 restarts, at most 250 evaluations per restart, seed 7.

| Method | Route | Valid | Cost | Seconds |
| --- | --- | --- | ---: | ---: |
| Brute force | 0 → 1 → 3 → 2 → 0 | Yes | 18 | 0.000035 |
| Nearest neighbor | 0 → 1 → 2 → 3 → 0 | Yes | 23 | 0.000010 |
| QAOA with postselection | 0 → 1 → 2 → 3 → 0 | Yes | 23 | 0.953554 |

- QAOA total valid probability: **1.308952%**.
- Selected bitstring probability: 0.362728% (before conditioning).
- Unfiltered most probable bitstring: **invalid**.
- Expected penalized energy: 90.221261.
- Mean route cost conditioned on validity: 21.412874.
- Objective evaluations across all restarts: 558.
- Best restart optimizer success: True; trust-region radius reached its lower bound.

Here QAOA's selected route is 5 distance units (27.78%) above the exact optimum.
It matches nearest-neighbor cost and takes much longer. Optimizer success
means its stopping condition was met; it does not certify a global optimum.
Low valid probability illustrates a limitation of this shallow circuit and
ordinary X mixer. No quantum advantage is demonstrated or claimed.

## Three-city example

Command: `python route_qaoa.py --cities 3`

4 qubits; the other settings are unchanged.

| Method | Route | Valid | Cost | Seconds |
| --- | --- | --- | ---: | ---: |
| Brute force | 0 → 1 → 2 → 0 | Yes | 9 | 0.000009 |
| Nearest neighbor | 0 → 1 → 2 → 0 | Yes | 9 | 0.000003 |
| QAOA with postselection | 0 → 2 → 1 → 0 | Yes | 9 | 0.418371 |

- QAOA total valid probability: **70.758637%**.
- Selected bitstring probability: 35.379319%.
- Unfiltered most probable bitstring: valid.
- Expected penalized energy: 14.999471.
- Mean route cost conditioned on validity: 9.000000.
- Objective evaluations across all restarts: 658.
- Best restart optimizer success: True; trust-region radius reached its lower bound.

Both symmetric three-city tours have cost 9. Matching the optimum here says
little about optimization quality.

## Interpreting these observations

These are actual single-run wall-clock observations, excluding import time.
QAOA includes construction, optimization, and exact probability inspection.
Timings vary between runs; probabilities and angle convergence may also vary
with library versions and floating-point behavior. Reverse symmetric tours
can exchange ordering under tiny numerical differences. No hardware, noise,
or sampled measurements were used. Postselection examines the full simulated
distribution and should not be confused with a guarantee of receiving a valid
route from one measurement.
