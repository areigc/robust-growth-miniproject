
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


d = 2

Sigma = np.array([[1.0, 0.3],
                  [0.3, 1.0]])

c_mat = np.array([[0.5, 0.1],
                  [0.1, 0.5]])

Sigma_t = torch.tensor(Sigma, dtype=torch.float32)
c_t     = torch.tensor(c_mat, dtype=torch.float32)
Sigma_inv_t = torch.inverse(Sigma_t)
L_Sigma = torch.linalg.cholesky(Sigma_t)


analytical_M = 0.5 * np.linalg.inv(Sigma)
analytical_g = 0.125 * np.trace(c_mat @ np.linalg.inv(Sigma))
print(f"Analytical g* = {analytical_g:.6f}")



#  Neural-network architectures

class AgentPhi(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_growth_functional(agent, x, c):
    x = x.detach().requires_grad_(True)
    phi = agent(x)
    grad_phi = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]

    # Tr(c H) = sum_{ij} c_{ij} d^2 phi / dx_i dx_j
    dd = x.shape[1]
    tr_c_hess = torch.zeros(x.shape[0], device=x.device)
    for i in range(dd):
        g_i = grad_phi[:, i]
        grad2_i = torch.autograd.grad(g_i.sum(), x, create_graph=True)[0]
        for j in range(dd):
            tr_c_hess = tr_c_hess + c[i, j] * grad2_i[:, j]

    quad = (grad_phi @ c * grad_phi).sum(dim=1)   # nabla phi^T c nabla phi
    return 0.5 * tr_c_hess - 0.5 * quad, grad_phi


class AdversaryB(nn.Module):

    def __init__(self, dim, Sig, c):
        super().__init__()
        self.dim = dim
        self.register_buffer('Sig_inv', torch.inverse(Sig))
        self.register_buffer('B_sym', 0.5 * c @ torch.inverse(Sig))
        n_skew = dim * (dim - 1) // 2
        self.alpha = nn.Parameter(torch.zeros(n_skew))

    def get_B(self):
        A = torch.zeros(self.dim, self.dim, device=self.alpha.device)
        idx = 0
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                A[i, j] = self.alpha[idx]
                A[j, i] = -self.alpha[idx]
                idx += 1
        S = A @ self.Sig_inv
        return self.B_sym + S




def estimate_g_stationary(agent, n_samples=4096):

    z = torch.randn(n_samples, d)
    x = z @ L_Sigma.T
    f, _ = compute_growth_functional(agent, x, c_t)
    return f.mean()


def estimate_g_with_drift(agent, B, n_samples=4096, dt=0.02, n_steps=200, n_paths=128):

    sigma_chol = torch.linalg.cholesky(c_t)
    sqrt_dt = float(np.sqrt(dt))

    # Start from stationary distribution
    x = torch.randn(n_paths, d) @ L_Sigma.T

    # Simulate and collect samples
    samples = []
    for step in range(n_steps):
        dW = torch.randn(n_paths, d) * sqrt_dt
        x = x + (-x @ B.T) * dt + dW @ sigma_chol.T
        if step >= n_steps // 4:  # skip burn-in
            samples.append(x.clone())

    X = torch.cat(samples, dim=0)  # (many, d)
    # subsample
    idx = torch.randperm(X.shape[0])[:n_samples]
    X_sub = X[idx]

    f, _ = compute_growth_functional(agent, X_sub, c_t)
    return f.mean()


#  Training

def train(n_outer=200, lr_agent=3e-3, lr_adv=1e-2, n_samples=2048):
    agent = AgentPhi(d, hidden=64)
    adversary = AdversaryB(d, Sigma_t, c_t)

    opt_agent = torch.optim.Adam(agent.parameters(), lr=lr_agent)
    opt_adv   = torch.optim.Adam(adversary.parameters(), lr=lr_adv)

    history_g = []

    for epoch in range(n_outer):
        # -- Adversary steps: minimise g --
        for _ in range(2):
            opt_adv.zero_grad()
            B = adversary.get_B()
            g = estimate_g_with_drift(agent, B, n_samples=n_samples,
                                       dt=0.02, n_steps=150, n_paths=64)
            g.backward()       # loss = g, adversary minimises
            opt_adv.step()

        # -- Agent steps: maximise g --
        for _ in range(3):
            opt_agent.zero_grad()
            B = adversary.get_B().detach()
            g = estimate_g_with_drift(agent, B, n_samples=n_samples,
                                       dt=0.02, n_steps=150, n_paths=64)
            (-g).backward()    # maximise g = minimise -g
            opt_agent.step()

        # -- Logging (no_grad for B, but agent needs grad for Hessian) --
        B_cur = adversary.get_B().detach()
        g_cur = estimate_g_stationary(agent, n_samples=4096)
        history_g.append(g_cur.item())

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:4d} | g = {history_g[-1]:.6f} | "
                  f"alpha = {adversary.alpha.data.numpy()} | "
                  f"g* = {analytical_g:.6f}")

    return agent, adversary, history_g



#  Plots


def make_plots(agent, adversary, history_g):
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Convergence
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history_g, lw=1, label='Learned $g$')
    ax.axhline(analytical_g, color='r', ls='--', lw=1,
               label=f'Analytical $g^*={analytical_g:.4f}$')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Growth rate')
    ax.set_title('Convergence of robust-optimal growth rate')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'growth_rate_convergence.png'), dpi=150)
    plt.close(fig)

    # 2. Strategy comparison (vector field)
    grid = np.linspace(-3, 3, 16)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    X_grid = torch.tensor(pts, dtype=torch.float32).requires_grad_(True)
    phi = agent(X_grid)
    grad_phi = torch.autograd.grad(phi.sum(), X_grid)[0].detach().numpy()

    Sinv = np.linalg.inv(Sigma)
    theta_star = 0.5 * pts @ Sinv.T

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, th, title in zip(axes,
                              [grad_phi, theta_star],
                              ['Learned $\\theta = \\nabla\\varphi$',
                               'Analytical $\\theta^* = \\frac{1}{2}\\Sigma^{-1}x$']):
        ax.quiver(pts[:, 0], pts[:, 1], th[:, 0], th[:, 1],
                  np.linalg.norm(th, axis=1), cmap='viridis')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'strategy_comparison.png'), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


# ==================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("  Robust Growth Rate - Adversarial Training")
    print("=" * 50)
    print(f"d = {d},  Analytical g* = {analytical_g:.6f}\n")

    agent, adversary, history_g = train(n_outer=200, lr_agent=3e-3,
                                         lr_adv=1e-2, n_samples=2048)

    B_learned = adversary.get_B().detach().numpy()
    print(f"\nLearned B =\n{B_learned}")
    print(f"B_sym (fixed) = (1/2) c Sigma^(-1) =\n{0.5 * c_mat @ np.linalg.inv(Sigma)}")
    print(f"Learned alpha = {adversary.alpha.data.numpy()}")
    print(f"\nFinal learned g = {history_g[-1]:.6f}")
    print(f"Analytical g*   = {analytical_g:.6f}")

    make_plots(agent, adversary, history_g)
