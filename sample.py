"""Inference script for PCFlow.

Samples a GPR B-scan from a gprMax-style condition (.in file) using a trained
checkpoint. Run from the repository root:

    python sample.py --ckpt outputs/gpr_cfm/checkpoints/ckpt_0050000.pt \
                     --infile path/to/scene.in --out result.png

Alternatively pass --jsonl to reuse an entry from a dataset split file.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.gpr_dataset.dataset import _read_png_gray
from data.gpr_dataset.parser_gprmax_in import parse_gprmax_in
from models.physic_encoder import EncoderNormSpec, MaxwellConditionEncoder
from models.unet import UNet_v
from models.vae_sdxl import SDXLVAE, VAEConfig
from utils.config import load_yaml
from utils.io import save_image_grid_rgb
from utils.solvers import euler_solve, heun_solve


def main():
    ap = argparse.ArgumentParser(description="Sample PCFlow from a gprMax .in condition.")
    ap.add_argument("--model-config", default="configs/model_gpr.yaml")
    ap.add_argument("--train-config", default="configs/train_gpr.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=False)
    ap.add_argument("--image", required=False)
    ap.add_argument("--infile", required=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    device = train_cfg["train"]["device"]

    # Pretrained VAE (encoder maps the pseudo-RGB image to the latent space).
    vae = SDXLVAE(
        VAEConfig(
            pretrained_path=model_cfg["vae"]["pretrained_path"],
            torch_dtype=model_cfg["vae"]["torch_dtype"],
            device=device,
            use_mean=model_cfg["vae"]["use_mean"],
        )
    )

    # Physics condition encoder.
    norm = EncoderNormSpec(**model_cfg["encoder"]["norm"])
    encoder = MaxwellConditionEncoder(
        grid_spacing=tuple(model_cfg["encoder"]["grid_spacing"]),
        freq=float(model_cfg["encoder"]["freq"]),
        max_layers=int(model_cfg["encoder"]["max_layers"]),
        max_interfaces=int(model_cfg["encoder"]["max_interfaces"]),
        norm=norm,
    ).to(device).eval()

    # Conditional rectified-flow UNet.
    unet = UNet_v(
        in_channels=model_cfg["model"]["in_channels"],
        cond_channels=model_cfg["model"]["cond_channels"],
        time_dim=model_cfg["model"]["time_dim"],
        base_channels=model_cfg["model"]["base_channels"],
        dropout=model_cfg["model"]["dropout"],
    ).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "ema" in ckpt:
        unet.load_state_dict(ckpt["ema"]["shadow"])
    else:
        unet.load_state_dict(ckpt["unet"])

    # Resolve one sample: either from a jsonl record or explicit files.
    if args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8") as f:
            rec = json.loads(next(f))
        img_path = rec["image_path"]
        in_path = rec["in_path"]
    else:
        img_path = args.image
        in_path = args.infile
    assert img_path and in_path

    H = W = train_cfg["data"]["image_size"]
    I_gray = _read_png_gray(img_path, (W, H)).unsqueeze(0).to(device)  # [1,1,H,W]

    with open(in_path, "r", encoding="utf-8") as f:
        in_obj = parse_gprmax_in(f.read())

    # Build the eps/sigma maps from the soil box material; this matches the
    # single-object dataset-generation convention.
    if len(in_obj.boxes) == 0:
        eps_r = 10.0
        sig = 0.01
    else:
        mat_name = in_obj.boxes[0].material
        mat = in_obj.materials.get(mat_name, None)
        eps_r = mat.eps_r if mat else 10.0
        sig = mat.sigma if mat else 0.01

    eps_map = torch.full((1, H, W), float(eps_r), device=device)
    sig_map = torch.full((1, H, W), float(sig), device=device)

    geom = {"layer_bounds": [], "source_pos": [], "pml_mask": None}

    with torch.no_grad():
        C_img = encoder(eps_map, sig_map, geom)  # [1,26,H,W]
        C_lat = F.avg_pool2d(C_img, kernel_size=8).to(torch.float16)  # [1,26,128,128]

        z0 = torch.randn(1, 4, 128, 128, device=device, dtype=torch.float16)
        v_fn = lambda z, t, c: unet(z, t, c)

        solver = train_cfg["solver"]["name"]
        steps = int(train_cfg["solver"]["steps"])
        if solver == "euler":
            z = euler_solve(v_fn, z0, C_lat, steps=steps)
        else:
            z = heun_solve(v_fn, z0, C_lat, steps=steps)

        x = vae.decode(z)
        save_image_grid_rgb(x, args.out)

    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
