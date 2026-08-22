#!/usr/bin/env python3
"""Analytical benchmark and adversarial training for robust growth.

The experiment learns a functionally generated trading strategy for a
two-dimensional Ornstein-Uhlenbeck market while an adversary changes the drift
subject to the stationary-covariance (Lyapunov) constraint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


DEFAULT_SEED = 42
DEFAULT_SIGMA = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float64)
DEFAULT_C = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.float64)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for alternating adversarial training."""

    epochs: int = 200
    agent_lr: float = 3e-3
    adversary_lr: float = 1e-2
    samples: int = 2048
    evaluation_samples: int = 4096
    simulation_dt: float = 0.02
    simulation_steps: int = 150
    paths: int = 64
    burn_in_fraction: float = 0.25
    agent_steps: int = 3
    adversary_steps: int = 2
    hidden: int = 64
    log_every: int = 25
    seed: int = DEFAULT_SEED


def seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible CPU runs."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def analytical_solution(
    sigma: np.ndarray = DEFAULT_SIGMA,
    c: np.ndarray = DEFAULT_C,
) -> tuple[np.ndarray, float]:
    """Return the optimal quadratic coefficient and closed-form growth rate."""

    sigma_inv = np.linalg.inv(sigma)
    optimal_m = 0.5 * sigma_inv
    optimal_g = 0.125 * np.trace(c @ sigma_inv)
    return optimal_m, float(optimal_g)


class AgentPhi(nn.Module):
    """Smooth generating function whose gradient is the trading strategy."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AdversaryB(nn.Module):
    """OU drift parameterisation satisfying B Sigma + Sigma B^T = c."""

    def __init__(self, sigma: torch.Tensor, c: torch.Tensor) -> None:
        super().__init__()
        if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
            raise ValueError("sigma must be a square matrix")
        self.dim = sigma.shape[0]
        sigma_inv = torch.linalg.inv(sigma)
        self.register_buffer("sigma_inv", sigma_inv)
        self.register_buffer("base_drift", 0.5 * c @ sigma_inv)
        self.alpha = nn.Parameter(
            torch.zeros(
                self.dim * (self.dim - 1) // 2,
                dtype=sigma.dtype,
                device=sigma.device,
            )
        )

    def get_B(self) -> torch.Tensor:
        skew = torch.zeros(
            self.dim,
            self.dim,
            dtype=self.alpha.dtype,
            device=self.alpha.device,
        )
        index = 0
        for row in range(self.dim):
            for column in range(row + 1, self.dim):
                skew[row, column] = self.alpha[index]
                skew[column, row] = -self.alpha[index]
                index += 1
        return self.base_drift + skew @ self.sigma_inv


def compute_growth_functional(
    agent: AgentPhi,
    x: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the pointwise growth integrand and strategy gradient."""

    # Keep an existing graph (notably the path from simulated states to the
    # adversary's drift), while making independent stationary samples
    # differentiable when needed for the spatial derivatives below.
    if not x.requires_grad:
        x = x.requires_grad_(True)
    phi = agent(x)
    grad_phi = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]

    trace_c_hessian = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for row in range(x.shape[1]):
        second_row = torch.autograd.grad(
            grad_phi[:, row].sum(), x, create_graph=True
        )[0]
        for column in range(x.shape[1]):
            trace_c_hessian = (
                trace_c_hessian + c[row, column] * second_row[:, column]
            )

    quadratic = (grad_phi @ c * grad_phi).sum(dim=1)
    return 0.5 * trace_c_hessian - 0.5 * quadratic, grad_phi


def estimate_g_stationary(
    agent: AgentPhi,
    c: torch.Tensor,
    sigma_cholesky: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Estimate growth from independent stationary Gaussian samples."""

    x = torch.randn(n_samples, sigma_cholesky.shape[0]) @ sigma_cholesky.T
    functional, _ = compute_growth_functional(agent, x, c)
    return functional.mean()


def estimate_g_with_drift(
    agent: AgentPhi,
    drift: torch.Tensor,
    c: torch.Tensor,
    sigma_cholesky: torch.Tensor,
    *,
    n_samples: int,
    dt: float,
    n_steps: int,
    n_paths: int,
    burn_in_fraction: float,
) -> torch.Tensor:
    """Estimate growth from differentiable Euler-Maruyama OU paths."""

    diffusion_cholesky = torch.linalg.cholesky(c)
    sqrt_dt = float(np.sqrt(dt))
    dim = sigma_cholesky.shape[0]

    x = torch.randn(n_paths, dim) @ sigma_cholesky.T
    samples: list[torch.Tensor] = []
    burn_in_steps = int(n_steps * burn_in_fraction)

    for step in range(n_steps):
        d_w = torch.randn(n_paths, dim) * sqrt_dt
        x = x + (-x @ drift.T) * dt + d_w @ diffusion_cholesky.T
        if step >= burn_in_steps:
            samples.append(x)

    if not samples:
        raise ValueError("burn-in consumes all simulated samples")

    simulated = torch.cat(samples, dim=0)
    count = min(n_samples, simulated.shape[0])
    indices = torch.randperm(simulated.shape[0])[:count]
    functional, _ = compute_growth_functional(agent, simulated[indices], c)
    return functional.mean()


def train(
    sigma: np.ndarray = DEFAULT_SIGMA,
    c: np.ndarray = DEFAULT_C,
    config: TrainConfig = TrainConfig(),
) -> tuple[AgentPhi, AdversaryB, list[float]]:
    """Run alternating minimax optimisation."""

    validate_config(config)
    seed_everything(config.seed)

    dim = sigma.shape[0]
    sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
    c_tensor = torch.tensor(c, dtype=torch.float32)
    sigma_cholesky = torch.linalg.cholesky(sigma_tensor)

    agent = AgentPhi(dim, hidden=config.hidden)
    adversary = AdversaryB(sigma_tensor, c_tensor)
    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=config.agent_lr)
    adversary_optimizer = torch.optim.Adam(
        adversary.parameters(), lr=config.adversary_lr
    )
    _, optimal_g = analytical_solution(sigma, c)
    history: list[float] = []

    estimator_arguments = {
        "n_samples": config.samples,
        "dt": config.simulation_dt,
        "n_steps": config.simulation_steps,
        "n_paths": config.paths,
        "burn_in_fraction": config.burn_in_fraction,
    }

    for epoch in range(config.epochs):
        for _ in range(config.adversary_steps):
            adversary_optimizer.zero_grad(set_to_none=True)
            agent.zero_grad(set_to_none=True)
            growth = estimate_g_with_drift(
                agent,
                adversary.get_B(),
                c_tensor,
                sigma_cholesky,
                **estimator_arguments,
            )
            growth.backward()
            adversary_optimizer.step()

        for _ in range(config.agent_steps):
            agent_optimizer.zero_grad(set_to_none=True)
            growth = estimate_g_with_drift(
                agent,
                adversary.get_B().detach(),
                c_tensor,
                sigma_cholesky,
                **estimator_arguments,
            )
            (-growth).backward()
            agent_optimizer.step()

        current_growth = estimate_g_stationary(
            agent, c_tensor, sigma_cholesky, config.evaluation_samples
        )
        history.append(current_growth.item())

        if (epoch + 1) % config.log_every == 0 or epoch + 1 == config.epochs:
            print(
                f"Epoch {epoch + 1:4d} | g = {history[-1]:.6f} | "
                f"alpha = {adversary.alpha.detach().numpy()} | g* = {optimal_g:.6f}"
            )

    return agent, adversary, history


def make_plots(
    agent: AgentPhi,
    history: Sequence[float],
    sigma: np.ndarray,
    c: np.ndarray,
    output_dir: Path,
) -> None:
    """Write convergence and strategy-comparison figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _, optimal_g = analytical_solution(sigma, c)

    figure, axis = plt.subplots(figsize=(5, 3))
    axis.plot(history, linewidth=1, label="Learned $g$")
    axis.axhline(
        optimal_g,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Analytical $g^*={optimal_g:.4f}$",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Growth rate")
    axis.set_title("Convergence of robust-optimal growth rate")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "growth_rate_convergence.png", dpi=150)
    plt.close(figure)

    grid = np.linspace(-3, 3, 16)
    x_grid, y_grid = np.meshgrid(grid, grid)
    points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    tensor_points = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    phi = agent(tensor_points)
    learned_strategy = torch.autograd.grad(phi.sum(), tensor_points)[0].numpy()
    analytical_strategy = 0.5 * points @ np.linalg.inv(sigma).T

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    titles = [
        "Learned $\\theta = \\nabla\\varphi$",
        "Analytical $\\theta^* = \\frac{1}{2}\\Sigma^{-1}x$",
    ]
    for axis, strategy, title in zip(
        axes, [learned_strategy, analytical_strategy], titles
    ):
        axis.quiver(
            points[:, 0],
            points[:, 1],
            strategy[:, 0],
            strategy[:, 1],
            np.linalg.norm(strategy, axis=1),
            cmap="viridis",
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("$x_1$")
        axis.set_ylabel("$x_2$")
        axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output_dir / "strategy_comparison.png", dpi=150)
    plt.close(figure)


def validate_config(config: TrainConfig) -> None:
    """Reject invalid simulation and optimisation settings early."""

    integer_fields = (
        "epochs",
        "samples",
        "evaluation_samples",
        "simulation_steps",
        "paths",
        "agent_steps",
        "adversary_steps",
        "hidden",
        "log_every",
    )
    for field in integer_fields:
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if config.agent_lr <= 0 or config.adversary_lr <= 0:
        raise ValueError("learning rates must be positive")
    if config.simulation_dt <= 0:
        raise ValueError("simulation_dt must be positive")
    if not 0 <= config.burn_in_fraction < 1:
        raise ValueError("burn_in_fraction must lie in [0, 1)")


def write_metrics(
    output_dir: Path,
    sigma: np.ndarray,
    c: np.ndarray,
    config: TrainConfig,
    adversary: AdversaryB,
    history: Sequence[float],
) -> None:
    """Save the key settings and results in a machine-readable file."""

    _, optimal_g = analytical_solution(sigma, c)
    metrics = {
        "sigma": sigma.tolist(),
        "c": c.tolist(),
        "training": asdict(config),
        "analytical_growth_rate": optimal_g,
        "final_learned_growth_rate": history[-1],
        "learned_alpha": adversary.alpha.detach().tolist(),
        "learned_drift": adversary.get_B().detach().tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a small smoke-test configuration instead of the full experiment",
    )
    parser.add_argument("--epochs", type=int, help="number of alternating epochs")
    parser.add_argument("--samples", type=int, help="samples per growth estimate")
    parser.add_argument(
        "--evaluation-samples", type=int, help="stationary samples used for logging"
    )
    parser.add_argument("--simulation-steps", type=int, help="OU steps per estimate")
    parser.add_argument("--paths", type=int, help="parallel OU paths per estimate")
    parser.add_argument("--agent-lr", type=float, default=3e-3)
    parser.add_argument("--adversary-lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build either the submitted or quick configuration with CLI overrides."""

    defaults = {
        "epochs": 3 if args.quick else 200,
        "samples": 256 if args.quick else 2048,
        "evaluation_samples": 512 if args.quick else 4096,
        "simulation_steps": 30 if args.quick else 150,
        "paths": 16 if args.quick else 64,
        "log_every": 1 if args.quick else 25,
    }
    return TrainConfig(
        epochs=args.epochs if args.epochs is not None else defaults["epochs"],
        samples=args.samples if args.samples is not None else defaults["samples"],
        evaluation_samples=(
            args.evaluation_samples
            if args.evaluation_samples is not None
            else defaults["evaluation_samples"]
        ),
        simulation_steps=(
            args.simulation_steps
            if args.simulation_steps is not None
            else defaults["simulation_steps"]
        ),
        paths=args.paths if args.paths is not None else defaults["paths"],
        agent_lr=args.agent_lr,
        adversary_lr=args.adversary_lr,
        log_every=defaults["log_every"],
        seed=args.seed,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    optimal_m, optimal_g = analytical_solution()

    print("=" * 58)
    print("Robust growth rate - adversarial training")
    print("=" * 58)
    print(f"Analytical g* = {optimal_g:.6f}")
    print(f"Analytical M* =\n{optimal_m}\n")

    agent, adversary, history = train(config=config)
    make_plots(agent, history, DEFAULT_SIGMA, DEFAULT_C, args.output_dir)
    write_metrics(
        args.output_dir,
        DEFAULT_SIGMA,
        DEFAULT_C,
        config,
        adversary,
        history,
    )

    print(f"\nLearned B =\n{adversary.get_B().detach().numpy()}")
    print(f"Learned alpha = {adversary.alpha.detach().numpy()}")
    print(f"Final learned g = {history[-1]:.6f}")
    print(f"Analytical g*   = {optimal_g:.6f}")
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
