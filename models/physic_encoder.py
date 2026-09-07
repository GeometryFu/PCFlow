import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderNormSpec:
    eps_r_min: float = 1.0
    eps_r_max: float = 80.0
    sigma_min: float = 0.0
    sigma_max: float = 0.05
    alpha_min: float = 0.0
    alpha_max: float = 5.0
    beta_min: float = 0.0
    beta_max: float = 200.0
    eta_min: float = 0.0
    eta_max: float = 1.0


def _clip_norm(x: torch.Tensor, vmin: float, vmax: float, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, vmin, vmax)
    return (x - vmin) / (vmax - vmin + eps)


def _safe_norm01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(2, x.ndim))
    xmin = x.amin(dim=dims, keepdim=True)
    xmax = x.amax(dim=dims, keepdim=True)
    return (x - xmin) / (xmax - xmin + eps)


class MaxwellConditionEncoder(nn.Module):
    """
    Single-pipeline 26-channel response-domain condition encoder.

    Channel layout:
      00 row_time_coord
      01 col_scan_coord
      02 eps_n
      03 sig_n
      04 eta_n
      05 beta_n
      06 alpha_n
      07 vertical_attenuation
      08 phi_sin
      09 phi_cos
      10 material_edge
      11 impedance_edge
      12 pipe_mask
      13 pipe_edge
      14 pipe_sdf
      15 pipe_nx
      16 pipe_ny
      17 pipe_radius_map
      18 pipe_center_x_map
      19 pipe_depth_map
      20 hyperbola_prior
      21 apex_prior
      22 travel_time_curve_norm
      23 signed_time_residual
      24 response_attenuation_prior
      25 target_strength_prior
    """

    def __init__(
        self,
        grid_spacing: Tuple[float, float] = (0.0025, 0.0025),
        freq: float = 500e6,
        max_layers: int = 5,
        max_interfaces: int = 4,
        norm: Optional[EncoderNormSpec] = None,
    ):
        super().__init__()
        self.dx, self.dy = grid_spacing
        self.freq = freq
        self.omega = 2 * math.pi * freq
        self.max_layers = max_layers
        self.max_interfaces = max_interfaces

        self.mu0 = 4 * math.pi * 1e-7
        self.eps0 = 8.854187817e-12
        self.c0 = 299792458.0
        self.norm = norm or EncoderNormSpec()

    @torch.no_grad()
    def forward(self, epsilon_r: torch.Tensor, sigma: torch.Tensor, geometry: Dict) -> torch.Tensor:
        assert epsilon_r.ndim == 3 and sigma.ndim == 3
        B, H, W = epsilon_r.shape
        device, dtype = epsilon_r.device, epsilon_r.dtype

        domain_x, domain_y = self._get_domain_xy(geometry, H, W)
        dy_img = float(domain_y) / max(H - 1, 1)

        row_coord = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        col_coord = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)

        eps_n = _clip_norm(epsilon_r, self.norm.eps_r_min, self.norm.eps_r_max).unsqueeze(1)
        sig_n = _clip_norm(sigma, self.norm.sigma_min, self.norm.sigma_max).unsqueeze(1)

        eps_abs = epsilon_r.unsqueeze(1) * self.eps0
        sig = sigma.unsqueeze(1)
        tan_delta = sig / (self.omega * eps_abs + 1e-12)
        term = torch.sqrt(1 + tan_delta ** 2)
        beta = self.omega * torch.sqrt(self.mu0 * eps_abs / 2.0) * torch.sqrt(term + 1.0)
        alpha = self.omega * torch.sqrt(self.mu0 * eps_abs / 2.0) * torch.sqrt(torch.clamp(term - 1.0, min=0.0))
        eta = 1.0 / torch.sqrt(epsilon_r.unsqueeze(1) + 1e-12)

        eta_n = _clip_norm(eta, self.norm.eta_min, self.norm.eta_max)
        beta_n = _clip_norm(beta, self.norm.beta_min, self.norm.beta_max)
        alpha_n = _clip_norm(alpha, self.norm.alpha_min, self.norm.alpha_max)

        decay_step = torch.exp(-alpha * dy_img)
        A_depth = torch.cumprod(decay_step, dim=2).clamp(1e-4, 1.0)
        phi_raw = torch.cumsum(beta * dy_img, dim=2)
        phi_sin = (torch.sin(phi_raw) + 1.0) / 2.0
        phi_cos = (torch.cos(phi_raw) + 1.0) / 2.0

        material_edge = _safe_norm01(self._sobel_mag(eps_n) + self._sobel_mag(sig_n))
        impedance_edge = _safe_norm01(self._sobel_mag(eta))

        pipe = self._extract_pipe_params(geometry, epsilon_r, sigma, domain_x, domain_y)
        pipe_maps = self._make_pipe_geometry_maps(pipe, B, H, W, domain_x, domain_y, device, dtype)
        response = self._make_response_priors(
            pipe, epsilon_r, sigma, material_edge, impedance_edge, geometry,
            B, H, W, domain_x, domain_y, device, dtype
        )

        C = torch.cat(
            [
                row_coord,                         # 00
                col_coord,                         # 01
                eps_n,                             # 02
                sig_n,                             # 03
                eta_n,                             # 04
                beta_n,                            # 05
                alpha_n,                           # 06
                A_depth,                           # 07
                phi_sin,                           # 08
                phi_cos,                           # 09
                material_edge,                     # 10
                impedance_edge,                    # 11
                pipe_maps["mask"],                 # 12
                pipe_maps["edge"],                 # 13
                pipe_maps["sdf"],                  # 14
                pipe_maps["nx"],                   # 15
                pipe_maps["ny"],                   # 16
                pipe_maps["radius_map"],           # 17
                pipe_maps["center_x_map"],         # 18
                pipe_maps["depth_map"],            # 19
                response["hyperbola_prior"],       # 20
                response["apex_prior"],            # 21
                response["travel_curve_norm"],     # 22
                response["signed_time_residual"],  # 23
                response["response_attenuation"],  # 24
                response["target_strength"],       # 25
            ],
            dim=1,
        )
        assert C.shape[1] == 26, C.shape
        return C

    def _get_domain_xy(self, geometry: Dict, H: int, W: int) -> Tuple[float, float]:
        meta = geometry.get("meta", geometry)
        domain = meta.get("domain", None)
        if domain is None:
            return float(W) * self.dx, float(H) * self.dy
        return float(domain[0]), float(domain[1])

    def _sobel_mag(self, x: torch.Tensor) -> torch.Tensor:
        dtype, device = x.dtype, x.device
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x_pad, kx)
        gy = F.conv2d(x_pad, ky)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def _extract_pipe_params(self, geometry: Dict, epsilon_r: torch.Tensor, sigma: torch.Tensor, domain_x: float, domain_y: float) -> Dict:
        meta = geometry.get("meta", geometry)
        cylinders = meta.get("cylinders", geometry.get("cylinders", []))

        if cylinders:
            cyl = max(cylinders, key=lambda c: float(c.get("radius", 0.0)))
            cx = 0.5 * (float(cyl["x1"]) + float(cyl["x2"]))
            cy = 0.5 * (float(cyl["y1"]) + float(cyl["y2"]))
            r = float(cyl["radius"])
            mat = str(cyl.get("material", "pipe"))
        else:
            eps0, sig0 = epsilon_r[0], sigma[0]
            bg_eps = self._estimate_background_scalar(eps0)
            bg_sig = self._estimate_background_scalar(sig0)
            mask = ((eps0 - bg_eps).abs() > 1e-3) | ((sig0 - bg_sig).abs() > 1e-5)
            ys, xs = torch.where(mask)
            if len(xs) == 0:
                cx, cy, r, mat = 0.5 * domain_x, 0.5 * domain_y, 0.05, "unknown"
            else:
                cx = float(xs.float().mean() / max(eps0.shape[1] - 1, 1) * domain_x)
                cy = float(domain_y - ys.float().mean() / max(eps0.shape[0] - 1, 1) * domain_y)
                area = float(mask.float().mean().item()) * domain_x * domain_y
                r = math.sqrt(max(area, 1e-8) / math.pi)
                mat = "estimated"

        tx = meta.get("tx", None)
        if tx is not None and len(tx) >= 4:
            surface_y = float(tx[2])
        else:
            surface_y = float(meta.get("soil_top_y", domain_y))

        depth = max(surface_y - cy, 1e-4)
        return {
            "cx": max(0.0, min(domain_x, cx)),
            "cy": max(0.0, min(domain_y, cy)),
            "radius": max(r, 1e-4),
            "material": mat,
            "surface_y": surface_y,
            "depth": depth,
        }

    def _estimate_background_scalar(self, x: torch.Tensor) -> float:
        vals = torch.round(x.flatten() * 1000.0) / 1000.0
        unique, counts = torch.unique(vals, return_counts=True)
        non_air = unique > 1.05
        if non_air.any():
            u, c = unique[non_air], counts[non_air]
            return float(u[torch.argmax(c)].item())
        return float(unique[torch.argmax(counts)].item())

    def _make_pipe_geometry_maps(self, pipe, B, H, W, domain_x, domain_y, device, dtype):
        xs = torch.linspace(0.0, domain_x, W, device=device, dtype=dtype).view(1, 1, 1, W)
        ys = torch.linspace(domain_y, 0.0, H, device=device, dtype=dtype).view(1, 1, H, 1)

        cx = torch.tensor(pipe["cx"], device=device, dtype=dtype)
        cy = torch.tensor(pipe["cy"], device=device, dtype=dtype)
        r = torch.tensor(pipe["radius"], device=device, dtype=dtype)

        dx = xs - cx
        dy = ys - cy
        dist = torch.sqrt(dx * dx + dy * dy + 1e-12)

        mask = (dist <= r).to(dtype).expand(B, 1, H, W)
        boundary_sigma = max(float(domain_x) / max(W - 1, 1), float(domain_y) / max(H - 1, 1)) * 2.0
        edge = torch.exp(-((dist - r).abs() ** 2) / (2.0 * boundary_sigma ** 2)).expand(B, 1, H, W)

        clip_m = max(float(r) * 4.0, 0.05)
        signed = torch.clamp((dist - r) / clip_m, -1.0, 1.0)
        sdf = ((signed + 1.0) / 2.0).expand(B, 1, H, W)

        nx = (dx / (dist + 1e-6)).expand(B, 1, H, W) * edge
        ny = (dy / (dist + 1e-6)).expand(B, 1, H, W) * edge

        radius_map = torch.full((B, 1, H, W), float(pipe["radius"]) / max(domain_y, 1e-6), device=device, dtype=dtype).clamp(0, 1)
        center_x_map = torch.full((B, 1, H, W), float(pipe["cx"]) / max(domain_x, 1e-6), device=device, dtype=dtype).clamp(0, 1)
        depth_map = torch.full((B, 1, H, W), float(pipe["depth"]) / max(domain_y, 1e-6), device=device, dtype=dtype).clamp(0, 1)

        return {"mask": mask, "edge": edge, "sdf": sdf, "nx": nx, "ny": ny, "radius_map": radius_map, "center_x_map": center_x_map, "depth_map": depth_map}

    def _make_response_priors(self, pipe, epsilon_r, sigma, material_edge, impedance_edge, geometry, B, H, W, domain_x, domain_y, device, dtype):
        meta = geometry.get("meta", geometry)
        time_window = float(meta.get("time_window", 40e-9))
        tx = meta.get("tx", None)
        rx = meta.get("rx", None)

        tx0 = float(tx[1]) if tx is not None and len(tx) >= 4 else 0.0
        rx0 = float(rx[0]) if rx is not None and len(rx) >= 3 else tx0
        rx_offset = rx0 - tx0

        scan_tx_x = torch.linspace(0.0, domain_x, W, device=device, dtype=dtype)
        scan_rx_x = torch.clamp(scan_tx_x + rx_offset, 0.0, domain_x)

        cx = torch.tensor(pipe["cx"], device=device, dtype=dtype)
        depth = torch.tensor(pipe["depth"], device=device, dtype=dtype)

        d_tx = torch.sqrt((scan_tx_x - cx) ** 2 + depth ** 2 + 1e-12)
        d_rx = torch.sqrt((scan_rx_x - cx) ** 2 + depth ** 2 + 1e-12)
        path_len = d_tx + d_rx

        soil_eps = self._estimate_background_scalar(epsilon_r[0])
        soil_sig = self._estimate_background_scalar(sigma[0])
        v = self.c0 / math.sqrt(max(soil_eps, 1.0))

        twtt = path_len / v
        curve_row = torch.clamp(twtt / max(time_window, 1e-12) * (H - 1), 0.0, float(H - 1))

        row_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
        curve = curve_row.view(1, 1, W)
        sigma_row = max(4.0, 0.008 * H)
        hyperbola = torch.exp(-((row_grid - curve) ** 2) / (2.0 * sigma_row ** 2)).view(1, 1, H, W).expand(B, 1, H, W)

        apex_col = torch.clamp(cx / max(domain_x, 1e-6) * (W - 1), 0.0, float(W - 1))
        apex_row = torch.clamp((2.0 * depth / v) / max(time_window, 1e-12) * (H - 1), 0.0, float(H - 1))
        col_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
        sx = max(4.0, float(pipe["radius"]) / max(domain_x, 1e-6) * W * 3.0)
        sy = max(4.0, 0.008 * H)
        apex = torch.exp(-((row_grid - apex_row) ** 2) / (2.0 * sy ** 2) - ((col_grid - apex_col) ** 2) / (2.0 * sx ** 2)).view(1, 1, H, W).expand(B, 1, H, W)

        curve_norm = (curve_row / max(H - 1, 1)).view(1, 1, 1, W).expand(B, 1, H, W)
        row_norm = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, 1, H, 1)
        signed_residual = (torch.clamp(row_norm - curve_norm[:, :, :1, :], -0.5, 0.5) + 0.5).expand(B, 1, H, W)

        alpha_soil = self._alpha_scalar(max(soil_eps, 1.0), max(soil_sig, 0.0))
        atten_col = torch.exp(-alpha_soil * path_len).view(1, 1, 1, W)
        spreading_col = (1.0 / (1.0 + path_len ** 2)).view(1, 1, 1, W)
        response_attenuation = _safe_norm01(hyperbola * atten_col * spreading_col)

        edge_strength = torch.clamp(0.5 * material_edge.amax(dim=(2, 3), keepdim=True) + 0.5 * impedance_edge.amax(dim=(2, 3), keepdim=True), 0.0, 1.0)
        target_strength = response_attenuation * edge_strength

        return {
            "hyperbola_prior": hyperbola.clamp(0, 1),
            "apex_prior": apex.clamp(0, 1),
            "travel_curve_norm": curve_norm.clamp(0, 1),
            "signed_time_residual": signed_residual.clamp(0, 1),
            "response_attenuation": response_attenuation.clamp(0, 1),
            "target_strength": target_strength.clamp(0, 1),
        }

    def _alpha_scalar(self, eps_r: float, sigma: float) -> float:
        eps_abs = eps_r * self.eps0
        tan_delta = sigma / (self.omega * eps_abs + 1e-12)
        term = math.sqrt(1.0 + tan_delta ** 2)
        return float(self.omega * math.sqrt(self.mu0 * eps_abs / 2.0) * math.sqrt(max(term - 1.0, 0.0)))
