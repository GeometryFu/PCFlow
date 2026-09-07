"""Parser for a minimal gprMax 2-D input-file dialect.

The generator stores every command line prefixed with ``#`` so that the raw
geometry/material description can live inside a comment block that gprMax
itself ignores. This parser strips the prefix and reads back the scene.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Material:
    eps_r: float
    sigma: float
    mu_r: float
    magnetic_loss: float
    name: str


@dataclass
class Box:
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    material: str


@dataclass
class Cylinder:
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    radius: float
    material: str


@dataclass
class GprMaxIn:
    domain: Tuple[float, float, float]
    dxyz: Tuple[float, float, float]
    time_window: float
    waveform: Tuple[str, float, float, str]
    tx: Tuple[str, float, float, float, str]
    rx: Tuple[float, float, float]
    src_steps: Tuple[float, float, float]
    rx_steps: Tuple[float, float, float]
    materials: Dict[str, Material]
    boxes: List[Box]
    cylinders: List[Cylinder]


def _parse_floats(parts: List[str]) -> List[float]:
    return [float(p) for p in parts]


# Built-in materials that gprMax always provides.
BUILTIN_MATERIALS = {
    "free_space": Material(eps_r=1.0, sigma=0.0, mu_r=1.0, magnetic_loss=0.0, name="free_space"),
    "pec": Material(eps_r=1.0, sigma=1.6e10, mu_r=1.0, magnetic_loss=0.0, name="pec"),
}


def parse_gprmax_in(text: str) -> GprMaxIn:
    domain = (0.0, 0.0, 0.0)
    dxyz = (0.0, 0.0, 0.0)
    time_window = 0.0
    waveform = ("", 0.0, 0.0, "")
    tx = ("", 0.0, 0.0, 0.0, "")
    rx = (0.0, 0.0, 0.0)
    src_steps = (0.0, 0.0, 0.0)
    rx_steps = (0.0, 0.0, 0.0)
    materials: Dict[str, Material] = {}
    boxes: List[Box] = []
    cylinders: List[Cylinder] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, rest = line[1:].split(":", 1)
        key = key.strip().lower()
        parts = rest.strip().split()

        if key == "domain":
            vals = _parse_floats(parts[:3])
            domain = (vals[0], vals[1], vals[2])
        elif key == "dx_dy_dz":
            vals = _parse_floats(parts[:3])
            dxyz = (vals[0], vals[1], vals[2])
        elif key == "time_window":
            time_window = float(parts[0])
        elif key == "waveform":
            waveform = (parts[0], float(parts[1]), float(parts[2]), parts[3])
        elif key == "hertzian_dipole":
            x, y, z = _parse_floats(parts[1:4])
            tx = (parts[0], x, y, z, parts[4])
        elif key == "rx":
            x, y, z = _parse_floats(parts[:3])
            rx = (x, y, z)
        elif key == "src_steps":
            x, y, z = _parse_floats(parts[:3])
            src_steps = (x, y, z)
        elif key == "rx_steps":
            x, y, z = _parse_floats(parts[:3])
            rx_steps = (x, y, z)
        elif key == "material":
            eps_r, sigma, mu_r, magnetic_loss = _parse_floats(parts[:4])
            name = parts[4]
            materials[name] = Material(eps_r, sigma, mu_r, magnetic_loss, name)
        elif key == "box":
            x0, y0, z0, x1, y1, z1 = _parse_floats(parts[:6])
            boxes.append(Box(x0, y0, z0, x1, y1, z1, parts[6]))
        elif key == "cylinder":
            # Format: x1 y1 z1 x2 y2 z2 radius material
            x1, y1, z1, x2, y2, z2, radius = _parse_floats(parts[:7])
            cylinders.append(Cylinder(x1, y1, z1, x2, y2, z2, radius, parts[7]))

    all_materials = {**BUILTIN_MATERIALS, **materials}
    return GprMaxIn(
        domain=domain,
        dxyz=dxyz,
        time_window=time_window,
        waveform=waveform,
        tx=tx,
        rx=rx,
        src_steps=src_steps,
        rx_steps=rx_steps,
        materials=all_materials,
        boxes=boxes,
        cylinders=cylinders,
    )
