"""Tiny fixed-start TSP: QAOA, exhaustive search, and nearest neighbor."""

import argparse
from itertools import combinations, permutations
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


DISTANCES = np.array([
    [0, 2, 3, 9],
    [2, 0, 4, 5],
    [3, 4, 0, 8],
    [9, 5, 8, 0],
], dtype=float)


def valid_route(route, n):
    return (route is not None and len(route) == n + 1
            and route[0] == route[-1] == 0
            and sorted(route[:-1]) == list(range(n)))


def route_cost(route, distances):
    if not valid_route(route, len(distances)):
        raise ValueError("Route must visit every city once and return to city 0")
    return float(sum(distances[a, b] for a, b in zip(route, route[1:])))


def brute_force(distances):
    n = len(distances)
    routes = ([0, *order, 0] for order in permutations(range(1, n)))
    return min(routes, key=lambda route: route_cost(route, distances))


def nearest_neighbor(distances):
    route, unvisited = [0], set(range(1, len(distances)))
    while unvisited:
        city = min(unvisited, key=lambda c: (distances[route[-1], c], c))
        route.append(city)
        unvisited.remove(city)
    return route + [0]


def make_qubo(distances):
    """Return upper-triangular Q and constant: E(x) = x.T @ Q @ x + c.

    x[city_index * m + position] says city_index + 1 is at that position.
    """
    m = len(distances) - 1
    q = np.zeros((m * m, m * m))
    # This feasible tour is an upper bound; nonnegative travel costs are assumed.
    penalty = route_cost(nearest_neighbor(distances), distances) + 1

    def add(a, b, value):
        q[min(a, b), max(a, b)] += value

    # Each city appears once, and each position contains one city.
    groups = ([[i * m + t for t in range(m)] for i in range(m)]
              + [[i * m + t for i in range(m)] for t in range(m)])
    for group in groups:
        for a in group:
            add(a, a, -penalty)
        for a, b in combinations(group, 2):
            add(a, b, 2 * penalty)

    for i in range(m):
        add(i * m, i * m, distances[0, i + 1])
        last = i * m + m - 1
        add(last, last, distances[i + 1, 0])
        for j in range(m):
            for t in range(m - 1):
                add(i * m + t, j * m + t + 1, distances[i + 1, j + 1])
    return q, 2 * m * penalty, penalty


def basis_bits(n_qubits):
    # Qiskit basis index uses qubit 0 as the least significant bit.
    return ((np.arange(2 ** n_qubits)[:, None]
             >> np.arange(n_qubits)) & 1)


def decode(bits, n):
    assignment = np.asarray(bits).reshape(n - 1, n - 1)
    if not (np.all(assignment.sum(axis=0) == 1)
            and np.all(assignment.sum(axis=1) == 1)):
        return None
    return [0, *(assignment.argmax(axis=0) + 1).tolist(), 0]


def qaoa_circuit(angles, q, scale):
    circuit = QuantumCircuit(len(q))
    circuit.h(range(len(q)))
    layers = len(angles) // 2
    for gamma, beta in zip(angles[:layers], angles[layers:]):
        # P and CP apply exp(-i gamma * E(x)/scale), ignoring the constant.
        for a, b in zip(*np.nonzero(q)):
            phase = -gamma * q[a, b] / scale
            if a == b:
                circuit.p(phase, int(a))
            else:
                circuit.cp(phase, int(a), int(b))
        # RX(2 beta) = exp(-i beta X).
        circuit.rx(2 * beta, range(len(q)))
    return circuit


def solve_qaoa(distances, layers=2, restarts=3, maxiter=250, seed=7):
    q, constant, scale = make_qubo(distances)
    bits = basis_bits(len(q))
    energies = np.einsum("bi,ij,bj->b", bits, q, bits) + constant

    def probabilities(angles):
        circuit = qaoa_circuit(angles, q, scale)
        return Statevector.from_instruction(circuit).probabilities()

    def objective(angles):
        return float(probabilities(angles) @ energies / scale)

    rng = np.random.default_rng(seed)
    best, evaluations = None, 0
    for _ in range(restarts):
        result = minimize(objective, rng.uniform(0, np.pi, 2 * layers),
                          method="COBYLA",
                          options={"maxiter": maxiter, "tol": 1e-5})
        evaluations += result.nfev
        if best is None or result.fun < best.fun:
            best = result

    probs = probabilities(best.x)
    routes = [decode(row, len(distances)) for row in bits]
    valid = np.array([route is not None for route in routes])
    # Select by probability only; do not use brute-force route costs to choose.
    selected = int(np.argmax(np.where(valid, probs, -1)))
    mass = float(probs[valid].sum())
    mode = int(np.argmax(probs))
    return {
        "route": routes[selected] if mass > 0 else None,
        "valid_probability": mass,
        "selected_probability": float(probs[selected]),
        "mode_valid": bool(valid[mode]),
        "expected_energy": float(probs @ energies),
        "conditional_cost": float(probs[valid] @ energies[valid] / mass)
                            if mass > 0 else None,
        "evaluations": evaluations,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", type=int, choices=[3, 4], default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.layers, args.restarts, args.maxiter) < 1:
        parser.error("layers, restarts, and maxiter must be positive")
    distances = DISTANCES[:args.cities, :args.cities]
    print(f"Cities: {args.cities}; qubits: {(args.cities - 1) ** 2}; "
          f"layers: {args.layers}; seed: {args.seed}")
    print("Distances:\n", distances)
    print(f"{'Method':<18} {'Route':<22} {'Valid':<6} {'Cost':>7} {'Seconds':>10}")
    for name, solver in [("Brute force", brute_force),
                         ("Nearest neighbor", nearest_neighbor),
                         ("QAOA (postselect)", lambda d: solve_qaoa(
                             d, args.layers, args.restarts, args.maxiter, args.seed))]:
        start = perf_counter()
        result = solver(distances)
        elapsed = perf_counter() - start
        route = result["route"] if isinstance(result, dict) else result
        cost = route_cost(route, distances) if route is not None else float("nan")
        print(f"{name:<18} {str(route):<22} {str(valid_route(route, args.cities)):<6} "
              f"{cost:7.2f} {elapsed:10.6f}")
    print(f"\nQAOA valid probability: {result['valid_probability']:.6%}")
    print(f"Selected route bitstring probability: {result['selected_probability']:.6%}")
    print(f"Unfiltered most-probable bitstring valid: {result['mode_valid']}")
    print(f"Expected penalized energy: {result['expected_energy']:.6f}")
    print(f"Mean cost conditioned on validity: {result['conditional_cost']:.6f}")
    print(f"Optimizer evaluations (all restarts): {result['evaluations']}")
    print(f"Best restart optimizer success: {result['optimizer_success']}")
    print(f"Optimizer message: {result['optimizer_message']}")
    print("Timings exclude imports; QAOA includes QUBO setup, optimization, and decoding.")
    print("Exact statevector simulation and postselection; no quantum advantage claimed.")


if __name__ == "__main__":
    main()
