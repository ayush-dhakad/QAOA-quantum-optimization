# QAOA-quantum-optimization

A small, readable Python example of the traveling salesperson problem (TSP).
Visit each city once and return to city 0, minimizing total distance. Choose
3 cities (4 qubits) or 4 cities (9 qubits). Distances are fixed, symmetric,
nonnegative numbers in arbitrary units; edit `DISTANCES` to experiment.

This is an educational classical simulation of a quantum algorithm. It makes
no quantum advantage claim. Exhaustive search is the sensible solution at this size.

## Run

Python 3.11+ is recommended; this project was tested with Python 3.14.7.
The supplied `.venv` is already installed on the machine where this was created.
From this project directory:

```sh
source .venv/bin/activate
python -m unittest -v
python route_qaoa.py
python route_qaoa.py --cities 3
```

To create a fresh environment after copying the project:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` gives compatible version ranges. `requirements-lock.txt`
records the exact installed versions, including indirect dependencies; use
`python -m pip install -r requirements-lock.txt` to reproduce those versions.
No IBM account, quantum hardware, Aer package, or API key is needed.

Optional controls:

```sh
python route_qaoa.py --cities 4 --layers 2 --restarts 3 --maxiter 250 --seed 7
```

`maxiter` limits objective evaluations per restart in COBYLA. The seed controls
initial angles. Larger settings may improve a result, but provide no guarantee.

## Binary encoding and QUBO

Fix the starting/ending city to 0. Let `m = n - 1` be the remaining cities and
positions. Binary variable `x[i,t] = 1` means city `i+1` occupies position `t`.
Flatten it as qubit `i*m + t`. Thus the encoding uses `(n-1)^2` qubits instead
of `n^2`. Every city must occur once, and every position must contain one city.

For distance matrix `d`, the travel term is

```text
T(x) = sum_i d[0,i+1] x[i,0]
     + sum_t=0..m-2 sum_i,j d[i+1,j+1] x[i,t] x[j,t+1]
     + sum_i d[i+1,0] x[i,m-1]
```

Add squared constraint penalties:

```text
E(x) = T(x)
     + A sum_i (sum_t x[i,t] - 1)^2
     + A sum_t (sum_i x[i,t] - 1)^2
```

This is a **QUBO**: quadratic unconstrained binary optimization. Because
`x^2 = x` for binary variables, expanding the squares gives only constant,
linear, and pair terms. `make_qubo` stores these in an upper-triangular matrix
`Q`, so `E(x) = x.T @ Q @ x + constant`; off-diagonal terms are counted once.

We set `A` to the nearest-neighbor tour cost plus 1. That tour is a feasible
upper bound. Any invalid assignment incurs at least one integer squared
violation, and travel terms are nonnegative, so its energy exceeds that upper
bound. Valid states have zero penalty and their energy equals tour cost.
This reasoning assumes nonnegative distances.

## QAOA in five steps

1. Apply Hadamard gates to start in a uniform superposition of all bitstrings.
2. Apply the cost unitary `exp(-i gamma H_C/A)`, where `H_C|x> = E(x)|x>`.
   A linear term `a*x[q]` uses `P(-gamma*a/A)` and a pair term `b*x[q]*x[r]`
   uses `CP(-gamma*b/A)`. The constant adds only a global phase and is omitted
   from the circuit. Equivalently, binary variables map to `(I-Z)/2` in an
   Ising Hamiltonian. Dividing by `A` changes the scale, not the minimizer.
3. Apply the X mixer `exp(-i beta sum_q X[q])` using `RX(2*beta)` on each qubit.
4. Repeat steps 2–3 for `layers` rounds. SciPy COBYLA adjusts the `2*layers`
   angles to minimize exact expected penalized energy. This is per-instance
   variational optimization, with no training dataset or learned model.
5. Use Qiskit's `Statevector.from_instruction` to inspect final probabilities.
   Decode valid assignments and select the **most probable valid bitstring**.

The ordinary X mixer can produce invalid routes. The output therefore reports
total valid probability and whether the unfiltered most probable bitstring is
valid. Selection explicitly conditions on validity (postselection); it does
not repair invalid assignments or select by minimum route cost. The two
directions of a symmetric tour are separate bitstrings, with separate
probabilities. Exact distribution inspection enumerates all `2^qubits`
states, so this implementation does not scale to large problems.

There are no sampled shots or noise here. On hardware, invalid samples would
be discarded; valid probability is the chance a shot survives. The selected
bitstring probability is its unconditional probability. Expected penalized
energy includes invalid assignments; mean cost conditioned on validity only
includes valid routes. Neither is the same statistic as the selected route cost.

## Baselines and runtime

Brute force tries all `(n-1)!` tours and gives the exact optimum. Nearest neighbor
repeatedly visits the closest unvisited city (ties use city number). For the
four-city table its tour costs 23, while the optimum costs 18. For 3 symmetric
cities both possible tour directions cost 9, so that case tests mechanics
rather than distinguishing route quality.

Each printed runtime is one wall-clock observation using `perf_counter`.
Imports are excluded. The QAOA time includes QUBO setup, state enumeration,
all optimizer restarts, final simulation, and decoding. Baseline times include
their searches. These tiny timings are not a performance benchmark; noise,
machine load, simulation overhead, and the exact-probability decoder matter.
COBYLA may stop at its evaluation budget; its status/message are printed.

See `RESULTS.md` for actual runs and installed versions.

## Files and tests

- `route_qaoa.py`: encoding, circuit, optimizer, baselines, and command-line demo.
- `test_route_qaoa.py`: five small tests using Python's built-in `unittest`.
- `requirements.txt` / `requirements-lock.txt`: dependencies.

Tests check known baseline costs and invalid routes, all 16/512 QUBO energies
against an independent constraint formula, feasible-state counts, circuit
cost phases and bit order, mixer behavior, and a short optimizer smoke run.
Tests do not assume QAOA always finds the optimum.

## API references

- [Qiskit Statevector](https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info.Statevector)
- [SciPy minimize / COBYLA](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html)
