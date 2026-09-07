from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import AutoencoderKL


@dataclass
class VAEConfig:
    # HuggingFace repo id of the SDXL VAE; a local directory also works.
    pretrained_path: str = "stabilityai/sdxl-vae"
    torch_dtype: str = "fp16"  # "fp16" or "bf16" or "fp32"
    device: str = "cuda"
    use_mean: bool = True  # use the posterior mean for stability


class SDXLVAE(torch.nn.Module):
    """
    Wrapper to encode/decode using SDXL VAE.

    Input images should be in [-1,1] range with shape [B,3,1024,1024].
    Output latent z shape: [B,4,128,128]
    """
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.torch_dtype]
        self.vae = AutoencoderKL.from_pretrained(cfg.pretrained_path, torch_dtype=dtype)
        self.vae.to(cfg.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # scaling factor used by diffusers pipelines
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.13025)
        self.use_mean = cfg.use_mean
        self._dtype = dtype
        self._device = cfg.device

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,1024,1024] in [-1,1]
        returns z: [B,4,128,128]
        """
        #x = x.to(device=self._device, dtype=self._dtype)
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mean if self.use_mean else posterior.sample()
        z = z * self.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B,4,128,128]
        returns x: [B,3,1024,1024] in [-1,1]
        """
        z = z / self.scaling_factor
        x = self.vae.decode(z).sample
        return x
