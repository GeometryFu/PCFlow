"""Training script for PCFlow (physics-conditioned rectified flow for GPR imaging).

Run from the repository root:

    python train.py
    python train.py --train-config configs/train_gpr_dataset_split.yaml
"""

import argparse
import os
import random
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data.gpr_dataset.dataset import GPRCFMDataset
from models.physic_encoder import EncoderNormSpec, MaxwellConditionEncoder
from models.pseudo_rgb import pseudo_rgb_from_gray
from models.unet import UNet_v
from models.vae_sdxl import SDXLVAE, VAEConfig
from utils.config import load_yaml
from utils.ema import EMA
from utils.io import save_gray_from_vae_rgb
from utils.solvers import euler_solve, heun_solve
from utils.train_monitor import CSVLogger, load_or_create_fixed_val_indices, plot_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MSELoss(nn.Module):
    """Plain MSE reconstruction loss.

    Returns a (total, mse, dummy) triple so it stays interface-compatible with
    physics-augmented losses used in ablations.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, v_pred, v_target, pixel_weight=None):
        diff_sq = (v_pred - v_target) ** 2
        if pixel_weight is not None:
            diff_sq = diff_sq * pixel_weight
        loss_mse = diff_sq.mean()
        zero_grad = torch.tensor(0.0, device=v_pred.device)
        return self.alpha * loss_mse, loss_mse, zero_grad


@torch.no_grad()
def build_cond_batch(encoder, eps_b, sig_b, geom_list, device, dtype=torch.float16):
    """Encode per-sample physics fields and downsample to latent resolution."""
    C_list = []
    for i in range(eps_b.shape[0]):
        C = encoder(
            eps_b[i : i + 1].to(device),
            sig_b[i : i + 1].to(device),
            geom_list[i],
        )  # [1,26,H,W]
        C_list.append(C)
    C_img = torch.cat(C_list, dim=0)  # [B,26,H,W]
    C_lat = downsample_cond(C_img, factor=8)  # [B,26,H/8,W/8]
    return C_lat


def downsample_cond(C_img: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Edge-preserving downsample for thin-layer GPR conditions.

    Channel groups:
      - MaxPool  (one-hot / strong signals): 1-5, 15-18, 22-24
      - AvgPool  (smooth physical fields):    0, 10-14, 19-21, 25
      - MinPool  (signed-distance channels):  6-9 (keep minima -> interface)
    """
    assert C_img.ndim == 4
    B, K, H, W = C_img.shape
    assert H % factor == 0 and W % factor == 0

    max_idx = [1, 2, 3, 4, 5, 15, 16, 17, 18, 22, 23, 24]
    avg_idx = [0, 10, 11, 12, 13, 14, 19, 20, 21, 25]
    dist_idx = [6, 7, 8, 9]

    C_avg = C_img[:, avg_idx]
    C_max = C_img[:, max_idx]
    C_dist = C_img[:, dist_idx]

    C_avg_ds = F.avg_pool2d(C_avg, kernel_size=factor)
    C_max_ds = F.max_pool2d(C_max, kernel_size=factor)
    # PyTorch has no min-pool, emulate with -max(-x): any interface pixel (0.0)
    # inside the window keeps the window at the interface after downsampling.
    C_dist_ds = -F.max_pool2d(-C_dist, kernel_size=factor)

    out = torch.empty(B, K, H // factor, W // factor, device=C_img.device, dtype=C_img.dtype)
    out[:, avg_idx] = C_avg_ds
    out[:, max_idx] = C_max_ds
    out[:, dist_idx] = C_dist_ds
    return out


@torch.no_grad()
def sample_once(
    unet,
    vae,
    encoder,
    batch,
    solver_name="heun",
    steps=20,
    device="cuda",
    cfg_scale=4.0,
):
    """Sample one batch with classifier-free guidance.

    cfg_scale > 1 strengthens conditioning; larger values give straighter,
    better localized pipes but can over-smooth background texture.
    """
    unet.eval()
    I_gray = batch["I_gray"].to(device)
    eps = batch["epsilon_r"]
    sig = batch["sigma"]
    geom_list = batch["geometry"]

    C_lat = build_cond_batch(encoder, eps, sig, geom_list, device=device, dtype=torch.float32)
    B = I_gray.shape[0]
    z0 = torch.randn(B, 4, 128, 128, device=device, dtype=torch.float32)

    C_null = torch.zeros_like(C_lat)

    def v_fn(z, t, c_cond):
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([t, t], dim=0)
        c_in = torch.cat([C_lat, C_null], dim=0)
        out = unet(z_in, t_in, c_in)
        v_out = out[0] if isinstance(out, (tuple, list)) else out
        v_cond, v_uncond = v_out.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    if solver_name == "euler":
        z = euler_solve(v_fn, z0, C_lat, steps=steps)
    else:
        z = heun_solve(v_fn, z0, C_lat, steps=steps)
    return vae.decode(z)


def setup_resume_training(train_cfg, unet, ema, optimizer, scaler, device) -> int:
    """Restore training from a checkpoint if enabled in the config."""
    resume_cfg = train_cfg["train"].get("resume", {})
    if not resume_cfg.get("enabled", False):
        print("Resume disabled; training from scratch.")
        return 0

    ckpt_path = resume_cfg.get("checkpoint_path", None)
    workdir = Path(train_cfg["output"]["workdir"])
    ckpt_dir = workdir / "checkpoints"

    if ckpt_path is None:
        if not ckpt_dir.exists():
            return 0
        ckpt_files = sorted(
            ckpt_dir.glob("ckpt_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
            reverse=True,
        )
        if not ckpt_files:
            return 0
        ckpt_path = ckpt_files[0]
    else:
        ckpt_path = ckpt_dir / ckpt_path
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return 0

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(checkpoint["unet"])
    ema.load_state_dict(checkpoint["ema"])
    ema.shadow.to(device)

    step = checkpoint["step"]
    print(f"Resumed from step {step}")
    return step


def parse_args():
    ap = argparse.ArgumentParser(description="Train PCFlow on GPR data.")
    ap.add_argument("--model-config", default="configs/model_gpr.yaml")
    ap.add_argument("--train-config", default="configs/train_gpr.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    device = train_cfg["train"]["device"]
    set_seed(train_cfg["train"]["seed"])

    cfg_dropout_prob = float(train_cfg["train"]["cfg_dropout_prob"])
    cfg_guidance_scale = float(train_cfg["solver"].get("cfg_scale", 4.0))

    workdir = Path(train_cfg["output"]["workdir"])
    (workdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (workdir / "samples").mkdir(parents=True, exist_ok=True)
    (workdir / "samples_fixed").mkdir(parents=True, exist_ok=True)
    (workdir / "curves").mkdir(parents=True, exist_ok=True)

    train_ds = GPRCFMDataset(
        train_cfg["data"]["train_jsonl"], image_size=train_cfg["data"]["image_size"]
    )
    val_ds = GPRCFMDataset(
        train_cfg["data"]["val_jsonl"], image_size=train_cfg["data"]["image_size"]
    )
    metrics_logger = CSVLogger(workdir / "metrics" / "train_log.csv")
    fixed_val_indices = load_or_create_fixed_val_indices(
        workdir / "metrics" / "fixed_val_indices.json",
        len(val_ds),
        num_samples=int(train_cfg["train"].get("fixed_sample_count", 8)),
        seed=int(train_cfg["train"]["seed"]) + 999,
    )

    def collate_fn(batch):
        return {
            "id": [b["id"] for b in batch],
            "I_gray": torch.stack([b["I_gray"] for b in batch], dim=0),  # [B,1,H,W]
            "epsilon_r": torch.stack([b["epsilon_r"] for b in batch], dim=0),
            "sigma": torch.stack([b["sigma"] for b in batch], dim=0),
            "geometry": [b["geometry"] for b in batch],
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=train_cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )

    @torch.no_grad()
    def compute_val_velocity_loss(max_batches=16):
        unet.eval()
        vals = []
        for bi, batch_v in enumerate(val_loader):
            if bi >= max_batches:
                break
            I_gray_v = batch_v["I_gray"].to(device)
            eps_v = batch_v["epsilon_r"].to(device)
            sig_v = batch_v["sigma"]
            geom_v = batch_v["geometry"]

            I_rgb_v = pseudo_rgb_from_gray(I_gray_v)
            I_rgb_v = torch.cat(
                [
                    I_rgb_v[:, 0:1] * 2 - 1,
                    torch.tanh(I_rgb_v[:, 1:2]),
                    torch.tanh(I_rgb_v[:, 2:3]),
                ],
                dim=1,
            ).to(torch.float32)

            z1_v = vae.encode(I_rgb_v)
            C_lat_v = build_cond_batch(
                encoder, eps_v, sig_v, geom_v, device=device, dtype=torch.float32
            )

            Bv = z1_v.shape[0]
            z0_v = torch.randn_like(z1_v)
            t_v = torch.rand(Bv, device=device, dtype=torch.float32)
            zt_v = (1 - t_v.view(Bv, 1, 1, 1)) * z0_v + t_v.view(Bv, 1, 1, 1) * z1_v
            v_target_v = z1_v - z0_v

            out = unet(zt_v, t_v, C_lat_v)
            v_pred_v = out[0] if isinstance(out, (tuple, list)) else out
            loss_v, mse_v, _ = criterion(v_pred_v, v_target_v, pixel_weight=None)
            vals.append((loss_v.item(), mse_v.item()))

        unet.train()
        if not vals:
            return None, None
        return (
            sum(x[0] for x in vals) / len(vals),
            sum(x[1] for x in vals) / len(vals),
        )

    vae = SDXLVAE(
        VAEConfig(
            pretrained_path=model_cfg["vae"]["pretrained_path"],
            torch_dtype=model_cfg["vae"]["torch_dtype"],
            device=device,
            use_mean=model_cfg["vae"]["use_mean"],
        )
    )

    norm = EncoderNormSpec(**model_cfg["encoder"]["norm"])
    encoder = MaxwellConditionEncoder(
        grid_spacing=tuple(model_cfg["encoder"]["grid_spacing"]),
        freq=float(model_cfg["encoder"]["freq"]),
        max_layers=int(model_cfg["encoder"]["max_layers"]),
        max_interfaces=int(model_cfg["encoder"]["max_interfaces"]),
        norm=norm,
    ).to(device).eval()

    unet = UNet_v(
        in_channels=model_cfg["model"]["in_channels"],
        cond_channels=model_cfg["model"]["cond_channels"],
        time_dim=model_cfg["model"]["time_dim"],
        base_channels=model_cfg["model"]["base_channels"],
        dropout=model_cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=float(train_cfg["train"]["lr"]),
        weight_decay=float(train_cfg["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["train"]["amp"]))
    ema = EMA(unet, decay=float(train_cfg["train"]["ema_decay"]))
    ema.shadow.to(device)

    criterion = MSELoss(alpha=1.0)
    step = setup_resume_training(train_cfg, unet, ema, optimizer, scaler, device)
    unet.train()

    while step < int(train_cfg["train"]["max_steps"]):
        for batch in train_loader:
            step += 1
            if step > int(train_cfg["train"]["max_steps"]):
                break

            I_gray = batch["I_gray"].to(device)  # [B,1,1024,1024]
            eps = batch["epsilon_r"].to(device)
            sig = batch["sigma"]
            geom_list = batch["geometry"]

            I_rgb = pseudo_rgb_from_gray(I_gray)
            I_rgb = torch.cat(
                [
                    I_rgb[:, 0:1] * 2 - 1,
                    torch.tanh(I_rgb[:, 1:2]),
                    torch.tanh(I_rgb[:, 2:3]),
                ],
                dim=1,
            ).to(torch.float32)

            with torch.no_grad():
                z1 = vae.encode(I_rgb)  # [B,4,128,128]
                C_lat = build_cond_batch(
                    encoder, eps, sig, geom_list, device=device, dtype=torch.float32
                )

            B = z1.shape[0]
            z0 = torch.randn_like(z1)
            t = torch.rand(B, device=device, dtype=torch.float32)
            zt = (1 - t.view(B, 1, 1, 1)) * z0 + t.view(B, 1, 1, 1) * z1
            v_target = z1 - z0

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(train_cfg["train"]["amp"])):
                # Classifier-free-guidance training: drop the condition with
                # probability cfg_dropout_prob so the model learns unconditional
                # generation as well.
                keep_mask = torch.bernoulli(
                    torch.zeros(B, device=device) + (1.0 - cfg_dropout_prob)
                )
                keep_mask_bc = keep_mask.view(B, 1, 1, 1)
                C_lat_in = C_lat * keep_mask_bc

                v_pred = unet(zt, t, C_lat_in)

                # Depth-wise linear compensation: deeper rows (weaker signals)
                # get a higher weight (1.0 -> 3.0 over the depth axis).
                H_dim = v_target.shape[2]
                depth_weight = (
                    torch.linspace(1.0, 3.0, H_dim, device=device).view(1, 1, H_dim, 1)
                )
                loss, _, _ = criterion(v_pred, v_target, pixel_weight=depth_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(unet)

            if step % int(train_cfg["train"]["log_every"]) == 0:
                print(f"Step={step} | Total={loss.item():.5f}")
                metrics_logger.log(
                    {
                        "step": step,
                        "train_loss": float(loss.item()),
                        "val_loss": "",
                        "val_mse": "",
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )

            if step % int(train_cfg["train"].get("val_every", 1000)) == 0:
                val_loss, val_mse = compute_val_velocity_loss(
                    max_batches=int(train_cfg["train"].get("val_batches", 16))
                )
                print(f"[VAL] Step={step} | ValLoss={val_loss:.5f} | ValMSE={val_mse:.5f}")
                metrics_logger.log(
                    {
                        "step": step,
                        "train_loss": "",
                        "val_loss": float(val_loss),
                        "val_mse": float(val_mse),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                plot_curves(
                    workdir / "metrics" / "train_log.csv",
                    workdir / "curves" / "loss_curve.png",
                )

            if step % int(train_cfg["train"]["sample_every"]) == 0:
                for idx in fixed_val_indices:
                    batch_val = collate_fn([val_ds[idx]])
                    print(f"Sampling fixed validation id: {batch_val['id']}")
                    x = sample_once(
                        ema.shadow.to(device),
                        vae,
                        encoder,
                        batch_val,
                        solver_name=train_cfg["solver"]["name"],
                        steps=int(train_cfg["solver"]["steps"]),
                        device=device,
                        cfg_scale=cfg_guidance_scale,
                    )
                    save_gray_from_vae_rgb(
                        x,
                        str(workdir / "samples_fixed" / f"step_{step:07d}_{idx}_ch0.png"),
                        channel=0,
                    )
                unet.train()

            if step % int(train_cfg["train"]["save_every"]) == 0:
                ckpt = {
                    "step": step,
                    "unet": unet.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                torch.save(ckpt, workdir / "checkpoints" / f"ckpt_{step:07d}.pt")
                print(f"Saved checkpoint at step {step}")

    print("done")


if __name__ == "__main__":
    main()
