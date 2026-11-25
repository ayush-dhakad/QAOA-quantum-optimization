import unittest
from itertools import permutations

import numpy as np
from qiskit.quantum_info import Statevector

from route_qaoa import (DISTANCES, basis_bits, brute_force, decode, make_qubo,
                        nearest_neighbor, qaoa_circuit, route_cost, solve_qaoa,
                        valid_route)


class RouteTests(unittest.TestCase):
    def test_baselines_and_invalid_routes(self):
        self.assertEqual(route_cost(brute_force(DISTANCES), DISTANCES), 18)
        self.assertEqual(route_cost(nearest_neighbor(DISTANCES), DISTANCES), 23)
        for route in (None, [0, 1, 1, 3, 0], [0, 1, 2, 3], [1, 2, 3, 0, 1]):
            self.assertFalse(valid_route(route, 4))
        with self.assertRaises(ValueError):
            route_cost([0, 1, 0], DISTANCES)

    def test_all_qubo_states_against_tsp_definition(self):
        # Independently compute travel and squared constraints for every state.
        for n in (3, 4):
            d = DISTANCES[:n, :n]
            m = n - 1
            q, constant, penalty = make_qubo(d)
            valid_count = 0
            optimum = route_cost(brute_force(d), d)
            for bits in basis_bits(m * m):
                x = bits.reshape(m, m)
                travel = sum(d[0, i + 1] * x[i, 0]
                             + d[i + 1, 0] * x[i, -1] for i in range(m))
                travel += sum(d[i + 1, j + 1] * x[i, t] * x[j, t + 1]
                              for i in range(m) for j in range(m)
                              for t in range(m - 1))
                violations = sum((x.sum(axis=0) - 1) ** 2)
                violations += sum((x.sum(axis=1) - 1) ** 2)
                energy = bits @ q @ bits + constant
                self.assertAlmostEqual(energy, travel + penalty * violations)
                route = decode(bits, n)
                if route is not None:
                    valid_count += 1
                    self.assertTrue(valid_route(route, n))
                    self.assertEqual(energy, route_cost(route, d))
                else:
                    self.assertGreater(energy, optimum)
            self.assertEqual(valid_count, len(list(permutations(range(1, n)))))

    def test_cost_phase_and_bit_order(self):
        q, constant, scale = make_qubo(DISTANCES[:3, :3])
        bits = basis_bits(len(q))
        energy_without_constant = np.einsum("bi,ij,bj->b", bits, q, bits)
        gamma = 0.73
        state = Statevector.from_instruction(qaoa_circuit([gamma, 0], q, scale))
        expected = np.exp(-1j * gamma * energy_without_constant / scale)
        expected /= np.sqrt(2 ** len(q))
        np.testing.assert_allclose(state.data, expected, atol=1e-12)

    def test_mixer_changes_cost_phases_into_probabilities(self):
        q, _, scale = make_qubo(DISTANCES[:3, :3])
        probs = Statevector.from_instruction(qaoa_circuit([0.73, 0.4], q, scale)).probabilities()
        self.assertAlmostEqual(float(probs.sum()), 1)
        self.assertGreater(float(np.ptp(probs)), 0.01)

    def test_small_optimizer_run(self):
        result = solve_qaoa(DISTANCES[:3, :3], layers=1, restarts=1, maxiter=30)
        self.assertTrue(valid_route(result["route"], 3))
        self.assertGreater(result["valid_probability"], 0)
        self.assertLessEqual(result["valid_probability"], 1)
        self.assertEqual(result["conditional_cost"], 9)


if __name__ == "__main__":
    unittest.main()
