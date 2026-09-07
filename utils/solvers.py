import torch


@torch.no_grad()
def euler_solve(v_fn, z0: torch.Tensor, cond: torch.Tensor, steps: int = 30):
    z = z0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0],), i * dt, device=z.device, dtype=z.dtype)
        v = v_fn(z, t, cond)
        z = z + dt * v
    return z


@torch.no_grad()
def heun_solve(v_fn, z0: torch.Tensor, cond: torch.Tensor, steps: int = 20):
    z = z0
    dt = 1.0 / steps
    for i in range(steps):
        t0 = torch.full((z.shape[0],), i * dt, device=z.device, dtype=z.dtype)
        v0 = v_fn(z, t0, cond)
        z_euler = z + dt * v0
        t1 = torch.full((z.shape[0],), (i + 1) * dt, device=z.device, dtype=z.dtype)
        v1 = v_fn(z_euler, t1, cond)
        z = z + dt * 0.5 * (v0 + v1)
    return z