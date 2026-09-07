from pathlib import Path
import numpy as np
from PIL import Image
import torch


def save_image_grid_rgb(x: torch.Tensor, path: str):
    """
    x: [B,3,H,W] in [-1,1]
    Saves first image only.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = x[0].detach().cpu().clamp(-1, 1)
    img = (img * 0.5 + 0.5).numpy()  # [0,1]
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_gray01(x: torch.Tensor, path: str):
    """
    x: [B,1,H,W] in [0,1]
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = x[0].detach().cpu().squeeze(0).clamp(0, 1).numpy()
    img = (img * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_gray_from_vae_rgb(x_rgb: torch.Tensor, path: str, channel: int = 0):
    """
    x_rgb: [B,3,H,W] in [-1,1]
    Save only one channel as grayscale (default ch0).
    """
    x0 = x_rgb[:, channel:channel+1].clamp(-1, 1)
    x0 = x0 * 0.5 + 0.5  # -> [0,1]
    save_gray01(x0, path)