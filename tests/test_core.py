"""Core mathematical checks for the robust-growth experiment."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robust_growth import (  # noqa: E402
    DEFAULT_C,
    DEFAULT_SIGMA,
    AgentPhi,
    AdversaryB,
    analytical_solution,
    estimate_g_with_drift,
)


class AnalyticalSolutionTests(unittest.TestCase):
    def test_closed_form_growth_rate(self) -> None:
        optimal_m, optimal_g = analytical_solution()

        np.testing.assert_allclose(
            optimal_m, 0.5 * np.linalg.inv(DEFAULT_SIGMA), rtol=1e-12, atol=1e-12
        )
        self.assertAlmostEqual(optimal_g, 0.12912087912087913, places=12)


class AdversaryConstraintTests(unittest.TestCase):
    def test_drift_satisfies_lyapunov_constraint(self) -> None:
        sigma = torch.tensor(DEFAULT_SIGMA, dtype=torch.float64)
        c = torch.tensor(DEFAULT_C, dtype=torch.float64)
        adversary = AdversaryB(sigma, c)

        with torch.no_grad():
            adversary.alpha.fill_(0.73)

        drift = adversary.get_B()
        residual = drift @ sigma + sigma @ drift.T - c

        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_simulated_objective_is_connected_to_adversary(self) -> None:
        torch.manual_seed(42)
        sigma = torch.tensor(DEFAULT_SIGMA, dtype=torch.float32)
        c = torch.tensor(DEFAULT_C, dtype=torch.float32)
        adversary = AdversaryB(sigma, c)
        agent = AgentPhi(dim=2, hidden=8)

        growth = estimate_g_with_drift(
            agent,
            adversary.get_B(),
            c,
            torch.linalg.cholesky(sigma),
            n_samples=32,
            dt=0.02,
            n_steps=6,
            n_paths=8,
            burn_in_fraction=0.25,
        )
        growth.backward()

        self.assertIsNotNone(adversary.alpha.grad)
        self.assertTrue(torch.isfinite(adversary.alpha.grad).all())


if __name__ == "__main__":
    unittest.main()
