import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.gpr_dataset.parser_gprmax_in import parse_gprmax_in, GprMaxIn


def _read_png_gray(path: str, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("L").resize(size, resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _rasterize_objects_to_maps(in_obj: GprMaxIn, out_hw: Tuple[int, int]):
    H, W = out_hw
    domain_x, domain_y, _ = in_obj.domain

    img_dx = domain_x / W
    img_dy = domain_y / H

    eps_map = torch.full((H, W), 1.0, dtype=torch.float32)
    sig_map = torch.zeros((H, W), dtype=torch.float32)

    def get_x_col(x_meter):
        return max(0, min(W, int(round(x_meter / img_dx))))

    def get_y_row(y_meter):
        return max(0, min(H, H - int(round(y_meter / img_dy))))

    all_objects = []
    for box in in_obj.boxes:
        mat = in_obj.materials.get(box.material)
        if mat:
            all_objects.append(("box", box, mat))
    for cyl in in_obj.cylinders:
        mat = in_obj.materials.get(cyl.material)
        if mat:
            all_objects.append(("cylinder", cyl, mat))

    for obj_type, obj, mat in all_objects:
        if obj_type == "box":
            x0, x1 = get_x_col(obj.x0), get_x_col(obj.x1)
            r_top, r_btm = get_y_row(obj.y1), get_y_row(obj.y0)
            if x1 > x0 and r_btm > r_top:
                eps_map[r_top:r_btm, x0:x1] = float(mat.eps_r)
                sig_map[r_top:r_btm, x0:x1] = float(mat.sigma)

        elif obj_type == "cylinder":
            xs = torch.linspace(0, domain_x, W, dtype=torch.float32)
            ys = torch.linspace(domain_y, 0, H, dtype=torch.float32)
            grid_X, grid_Y = torch.meshgrid(xs, ys, indexing="xy")
            cx = (obj.x1 + obj.x2) / 2
            cy = (obj.y1 + obj.y2) / 2
            mask = torch.sqrt((grid_X - cx) ** 2 + (grid_Y - cy) ** 2) <= obj.radius
            eps_map[mask] = float(mat.eps_r)
            sig_map[mask] = float(mat.sigma)

    _, tx_x, tx_y, _, _ = in_obj.tx
    src_px, src_py = get_x_col(tx_x), get_y_row(tx_y)

    cylinders = [
        {
            "x1": c.x1,
            "y1": c.y1,
            "z1": c.z1,
            "x2": c.x2,
            "y2": c.y2,
            "z2": c.z2,
            "radius": c.radius,
            "material": c.material,
        }
        for c in in_obj.cylinders
    ]
    boxes = [
        {
            "x0": b.x0,
            "y0": b.y0,
            "z0": b.z0,
            "x1": b.x1,
            "y1": b.y1,
            "z1": b.z1,
            "material": b.material,
        }
        for b in in_obj.boxes
    ]
    materials = {
        name: {
            "eps_r": m.eps_r,
            "sigma": m.sigma,
            "mu_r": m.mu_r,
            "magnetic_loss": m.magnetic_loss,
            "name": m.name,
        }
        for name, m in in_obj.materials.items()
    }

    soil_top_y = max([b.y1 for b in in_obj.boxes], default=tx_y)
    y_tops = sorted({b.y1 for b in in_obj.boxes})
    layer_bounds = [y for y in y_tops if (y > 1e-9 and y < domain_y - 1e-9)]

    geom = {
        "layer_bounds": layer_bounds,
        "source_pos": [(src_px, src_py)],
        "pml_mask": None,
        "meta": {
            "domain": in_obj.domain,
            "dxyz": in_obj.dxyz,
            "time_window": in_obj.time_window,
            "waveform": in_obj.waveform,
            "tx": in_obj.tx,
            "rx": in_obj.rx,
            "src_steps": in_obj.src_steps,
            "rx_steps": in_obj.rx_steps,
            "num_boxes": len(in_obj.boxes),
            "num_cylinders": len(in_obj.cylinders),
            "boxes": boxes,
            "cylinders": cylinders,
            "materials": materials,
            "soil_top_y": soil_top_y,
        },
    }
    return eps_map, sig_map, geom


class GPRCFMDataset(Dataset):
    def __init__(self, jsonl_path: str, image_size: int = 1024):
        self.items: List[Dict[str, Any]] = []
        self.jsonl_path = Path(jsonl_path)
        self.image_size = image_size
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img_path = it["image_path"]
        in_path = it["in_path"]
        H = W = self.image_size

        I_gray = _read_png_gray(img_path, (W, H))
        with open(in_path, "r", encoding="utf-8") as f:
            in_obj = parse_gprmax_in(f.read())

        eps_map, sig_map, geom = _rasterize_objects_to_maps(in_obj, (H, W))
        return {
            "id": it.get("id", str(idx)),
            "I_gray": I_gray,
            "epsilon_r": eps_map,
            "sigma": sig_map,
            "geometry": geom,
        }
