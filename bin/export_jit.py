import os
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import OmegaConf, open_dict
from torch import nn

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers


class JITWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, mask):
        batch = {
            "image": image,
            "mask": mask
        }
        out = self.model(batch)
        return out["inpainted"]


@hydra.main(config_path="../configs/export", config_name="default.yaml")
def main(config: OmegaConf):
    register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

    checkpoint = torch.load(config.path)
    model_config = checkpoint["hyper_parameters"]
    with open_dict(model_config):
        model_config.training_model.predict_only = True
        model_config.visualizer.kind = "noop"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_checkpoint(
        model_config, config.path, strict=False, map_location=device
    )
    model.to(device)
    model.eval()
    jit_model_wrapper = JITWrapper(model)

    image = torch.rand(1, 3, 120, 120)
    mask = torch.rand(1, 1, 120, 120)
    image = image.to(device)
    mask = mask.to(device)

    output = jit_model_wrapper(image, mask)

    traced_model = torch.jit.trace(jit_model_wrapper, (image, mask), strict=False).to(device)

    save_path = Path(config.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving big-lama.pt model to {save_path}")
    traced_model.save(save_path)

    print(f"Checking jit model output...")
    jit_model = torch.jit.load(str(save_path), map_location=device)
    jit_output = jit_model(image, mask)
    diff = (output - jit_output).abs().sum()
    print(f"diff: {diff}")


if __name__ == "__main__":
    main()
