"""Differentiable 1D Richards equation PDE forward solver.

Features:
- Finite-volume cell-centered vertical discretization with ghost cells.
- Fully implicit backward Euler time stepping.
- Hydraulic-head upwinding for relative permeability.
- Inverse-distance harmonic mean for inter-cell saturated conductivity.
- Newton-Raphson iteration with Armijo backtracking line search.
- Tridiagonal and autograd Jacobian options.
- Differentiable with respect to constitutive parameters (latent z and scales).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Union

import torch

from .grid import Grid1D
from .material import Material, _as_tensor

TensorLike = Union[float, int, torch.Tensor]


@dataclass(frozen=True)
class BoundaryConditions:
    """Lower Dirichlet pressure head and upper boundary flux."""

    psi_lb: TensorLike | Callable[[float], TensorLike]
    q_ub: TensorLike | Callable[[float], TensorLike]

    def bottom_psi(self, t: float, grid: Grid1D) -> torch.Tensor:
        value = self.psi_lb(t) if callable(self.psi_lb) else self.psi_lb
        return _as_tensor(value, grid.z)

    def top_flux(self, t: float, grid: Grid1D) -> torch.Tensor:
        value = self.q_ub(t) if callable(self.q_ub) else self.q_ub
        return _as_tensor(value, grid.z)


@dataclass(frozen=True)
class NewtonOptions:
    tol: float = 1.0e-8
    max_iter: int = 15
    create_graph: bool = True
    jacobian_mode: str = "tridiagonal"  # "tridiagonal" or "dense_autograd"
    linear_solve_ridge: float = 0.0
    early_stop: bool = True
    max_line_search_iter: int = 8
    line_search_alpha: float = 1.0e-4
    damping: float = 1.0


@dataclass
class SimulationResult:
    times: torch.Tensor
    grid: Grid1D
    psi: torch.Tensor           # [N_time, Ns] interior pressure head
    theta: torch.Tensor         # [N_time, Ns] interior volumetric water content
    converged: bool
    final_residual: float


def hydraulic_head(psi: torch.Tensor, grid: Grid1D) -> torch.Tensor:
    return psi + grid.z


def face_harmonic_mean(cell_values: torch.Tensor, grid: Grid1D) -> torch.Tensor:
    left = cell_values[:-1]
    right = cell_values[1:]
    distance = grid.z[1:] - grid.z[:-1]
    denom = (grid.dz[:-1] * right + grid.dz[1:] * left) * 0.5
    denom = denom.clamp_min(torch.finfo(cell_values.dtype).eps)
    return left * right * distance / denom


def compute_fluxes(
    psi: torch.Tensor,
    grid: Grid1D,
    constitutive,
    material: Material,
) -> torch.Tensor:
    """Compute interface Darcy fluxes between adjacent cells."""
    ks_faces = face_harmonic_mean(material.saturated_conductivity(grid), grid)
    head = hydraulic_head(psi, grid)
    distance = grid.z[1:] - grid.z[:-1]
    head_gradient = (head[1:] - head[:-1]) / distance

    # Hydraulic-head upwinding
    use_left = head[:-1] >= head[1:]
    kr_cells = constitutive.kr(psi)
    kr_face = torch.where(use_left, kr_cells[:-1], kr_cells[1:])
    return -ks_faces * kr_face * head_gradient


def richards_residual(
    psi_new: torch.Tensor,
    psi_old: torch.Tensor,
    t_new: float,
    dt: TensorLike,
    grid: Grid1D,
    constitutive,
    material: Material,
    boundary: BoundaryConditions,
) -> torch.Tensor:
    """Finite-volume implicit backward Euler residual vector."""
    theta_new = constitutive.theta(psi_new)
    theta_old = constitutive.theta(psi_old)
    fluxes = compute_fluxes(psi_new, grid, constitutive, material)

    # Lower boundary face interpolation
    z_b = -grid.Z
    z0 = grid.z[0]
    z1 = grid.z[1]
    psi_face_bottom = ((z_b - z0) * psi_new[1] + (z1 - z_b) * psi_new[0]) / (z1 - z0)
    bottom = (psi_face_bottom - boundary.bottom_psi(t_new, grid)).reshape(1)

    # Interior cells mass conservation
    dt_tensor = _as_tensor(dt, grid.z)
    div_flux = (fluxes[:-1] - fluxes[1:]) / grid.dz_internal
    interior = theta_new[1:-1] - theta_old[1:-1] - dt_tensor * div_flux

    # Upper boundary face flux
    top = (fluxes[-1] - boundary.top_flux(t_new, grid)).reshape(1)
    return torch.cat([bottom, interior, top])


def solve_tridiagonal(
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Thomas algorithm for tridiagonal systems with pure out-of-place assembly."""
    n = int(diag.numel())
    if n == 1:
        return rhs / diag

    c_values = [upper[0] / diag[0]]
    d_values = [rhs[0] / diag[0]]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_values[-1]
        d_i = (rhs[i] - lower[i - 1] * d_values[-1]) / denom
        d_values.append(d_i)
        if i < n - 1:
            c_values.append(upper[i] / denom)

    x_values = [d_values[-1]]
    for i in range(n - 2, -1, -1):
        x_i = d_values[i] - c_values[i] * x_values[-1]
        x_values.append(x_i)
    x_values.reverse()
    return torch.stack(x_values)


def richards_tridiagonal_jacobian(
    psi_new: torch.Tensor,
    grid: Grid1D,
    constitutive,
    material: Material,
    dt: TensorLike,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytical tridiagonal Jacobian assembly for Richards residual."""
    psi_for_grad = psi_new if (create_graph and psi_new.requires_grad) else psi_new.detach().requires_grad_(True)
    theta = constitutive.theta(psi_for_grad)
    dtheta = torch.autograd.grad(theta.sum(), psi_for_grad, create_graph=create_graph, retain_graph=True)[0]

    psi_for_grad_kr = psi_new if (create_graph and psi_new.requires_grad) else psi_new.detach().requires_grad_(True)
    kr = constitutive.kr(psi_for_grad_kr)
    dkr = torch.autograd.grad(kr.sum(), psi_for_grad_kr, create_graph=create_graph, retain_graph=True)[0]

    ks_faces = face_harmonic_mean(material.saturated_conductivity(grid), grid)
    head = hydraulic_head(psi_new, grid)
    distance = grid.z[1:] - grid.z[:-1]
    head_grad = (head[1:] - head[:-1]) / distance
    use_left = head[:-1] >= head[1:]

    kr_left = kr[:-1]
    kr_right = kr[1:]
    dkr_left = dkr[:-1]
    dkr_right = dkr[1:]
    kr_face = torch.where(use_left, kr_left, kr_right)

    dflux_dleft = torch.where(
        use_left,
        -ks_faces * dkr_left * head_grad + ks_faces * kr_face / distance,
        ks_faces * kr_face / distance,
    )
    dflux_dright = torch.where(
        use_left,
        -ks_faces * kr_face / distance,
        -ks_faces * dkr_right * head_grad - ks_faces * kr_face / distance,
    )

    # Lower boundary face condition
    z_b = -grid.Z
    z0 = grid.z[0]
    z1 = grid.z[1]
    diag_0 = ((z1 - z_b) / (z1 - z0)).reshape(1)
    upper_0 = ((z_b - z0) / (z1 - z0)).reshape(1)

    # Interior cells
    dt_t = _as_tensor(dt, grid.z)
    scale = dt_t / grid.dz_internal
    lower_int = -scale * dflux_dleft[:-1]
    diag_int = dtheta[1:-1] - scale * (dflux_dright[:-1] - dflux_dleft[1:])
    upper_int = scale * dflux_dright[1:]

    # Upper boundary flux condition
    lower_top = dflux_dleft[-1].reshape(1)
    diag_top = dflux_dright[-1].reshape(1)

    lower = torch.cat([lower_int, lower_top], dim=0)
    diag = torch.cat([diag_0, diag_int, diag_top], dim=0)
    upper = torch.cat([upper_0, upper_int], dim=0)

    if not create_graph:
        lower = lower.detach()
        diag = diag.detach()
        upper = upper.detach()
    return lower, diag, upper


def newton_line_search_solve(
    x0: torch.Tensor,
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    options: NewtonOptions,
    jacobian_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, float, bool]:
    """Solve one nonlinear time step with Newton-Raphson & Armijo backtracking."""
    x = x0.clone()
    converged = False
    res_norm = float("inf")

    for _ in range(options.max_iter):
        f = residual_fn(x)
        with torch.no_grad():
            res_norm = float(torch.linalg.vector_norm(f.detach(), ord=float("inf")).cpu())
        if options.early_stop and res_norm <= options.tol:
            converged = True
            break

        if options.jacobian_mode == "tridiagonal" and jacobian_fn is not None:
            lower, diag, upper = jacobian_fn(x)
            if options.linear_solve_ridge:
                diag = diag + options.linear_solve_ridge
            direction = solve_tridiagonal(lower, diag, upper, -f)
        else:
            jac = torch.autograd.functional.jacobian(
                residual_fn,
                x,
                create_graph=options.create_graph,
                strict=False,
            )
            if options.linear_solve_ridge:
                jac = jac + options.linear_solve_ridge * torch.eye(jac.shape[0], dtype=jac.dtype, device=jac.device)
            direction = torch.linalg.solve(jac, -f)

        # Line search
        step_size = options.damping
        candidate = x + step_size * direction
        candidate_norm = float(torch.linalg.vector_norm(residual_fn(candidate).detach(), ord=float("inf")).cpu())

        ls_iter = 0
        while (
            candidate_norm > res_norm - res_norm * options.line_search_alpha * step_size
            and ls_iter < options.max_line_search_iter
        ):
            step_size *= 0.5
            candidate = x + step_size * direction
            candidate_norm = float(torch.linalg.vector_norm(residual_fn(candidate).detach(), ord=float("inf")).cpu())
            ls_iter += 1

        x = candidate

    if not converged:
        with torch.no_grad():
            res_norm = float(torch.linalg.vector_norm(residual_fn(x).detach(), ord=float("inf")).cpu())
        converged = res_norm <= options.tol

    return x, res_norm, converged


def initialize_psi(
    grid: Grid1D,
    constitutive,
    material: Material,
    boundary: BoundaryConditions,
    initial_psi: TensorLike | Iterable[float] | Callable[[torch.Tensor], torch.Tensor],
    t0: float = 0.0,
) -> torch.Tensor:
    """Initialize pressure head vector including ghost cells."""
    if callable(initial_psi):
        interior = _as_tensor(initial_psi(grid.z_internal), grid.z)
    else:
        values = _as_tensor(initial_psi, grid.z)
        interior = values.expand(grid.Ns) if values.ndim == 0 else values

    psi_lb = boundary.bottom_psi(t0, grid)
    z_b = -grid.Z
    z0 = grid.z[0]
    z1 = grid.z[1]
    bottom_ghost = ((psi_lb * (z1 - z0)) - (z_b - z0) * interior[0]) / (z1 - z_b)
    top_ghost = interior[-1]  # initial zero-flux ghost guess

    return torch.cat([bottom_ghost.reshape(1), interior, top_ghost.reshape(1)])


def solve_richards(
    grid: Grid1D,
    constitutive,
    material: Material,
    boundary: BoundaryConditions,
    initial_psi: TensorLike | Callable[[torch.Tensor], torch.Tensor],
    T: float,
    dt: float,
    options: NewtonOptions | None = None,
) -> SimulationResult:
    """Solve 1D transient Richards equation across duration T with time step dt."""
    options = options or NewtonOptions()
    n_steps = int(round(T / dt))
    times = torch.linspace(0.0, T, n_steps + 1, dtype=grid.z.dtype, device=grid.z.device)

    psi0 = initialize_psi(grid, constitutive, material, boundary, initial_psi, t0=float(times[0]))
    current_psi = psi0

    psi_history = [psi0[1:-1]]
    theta_history = [constitutive.theta(psi0[1:-1])]
    all_converged = True
    last_res = 0.0

    for step in range(1, n_steps + 1):
        t_new = float(times[step])
        psi_old = current_psi

        def step_res(x: torch.Tensor) -> torch.Tensor:
            return richards_residual(x, psi_old, t_new, dt, grid, constitutive, material, boundary)

        def step_jac(x: torch.Tensor):
            return richards_tridiagonal_jacobian(x, grid, constitutive, material, dt, create_graph=options.create_graph)

        current_psi, res_norm, step_conv = newton_line_search_solve(
            current_psi,
            step_res,
            options,
            jacobian_fn=step_jac if options.jacobian_mode == "tridiagonal" else None,
        )
        last_res = res_norm
        if not step_conv:
            all_converged = False

        psi_history.append(current_psi[1:-1])
        theta_history.append(constitutive.theta(current_psi[1:-1]))

    return SimulationResult(
        times=times,
        grid=grid,
        psi=torch.stack(psi_history, dim=0),
        theta=torch.stack(theta_history, dim=0),
        converged=all_converged,
        final_residual=last_res,
    )
