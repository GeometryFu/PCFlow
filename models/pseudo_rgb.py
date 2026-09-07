import torch


def pseudo_rgb_from_gray(x: torch.Tensor) -> torch.Tensor:
    """Convert a single-channel GPR B-scan into the 3-channel VAE input.

    x: [B,1,H,W] in [0,1]  ->  returns [B,3,H,W].

    The grayscale channel is replicated three times: edge/texture variants
    (Sobel/DoG) were explored during development but plain replication gave
    the best reconstruction fidelity, so it is the final convention.
    """
    return torch.cat([x, x, x], dim=1)
