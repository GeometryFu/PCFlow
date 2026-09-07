import copy
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, d):
        self.decay = d["decay"]
        self.shadow.load_state_dict(d["shadow"])