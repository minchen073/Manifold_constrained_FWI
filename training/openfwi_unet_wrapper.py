"""
70×70 denoising network: ``OpenFWIUNetWrapper`` is the model (input/output spatial×spatial); it contains an
inner ``UNet2DModel`` at sample_size ``spatial+2``.

Checkpoints: single ``model.pt`` via ``save_openfwi_checkpoint`` / ``load_openfwi_checkpoint`` (plain ``torch.save``).

If importing this module directly, call ``CurveVelB.diffusers_torch_compat.ensure_diffusers_custom_ops_safe()``
before the first ``import diffusers``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput


class OpenFWIUNetWrapper(nn.Module):
    def __init__(self, unet: UNet2DModel, *, spatial: int = 70, pad_mode: str = "replicate") -> None:
        super().__init__()
        inner = int(unet.config.sample_size)
        if inner != spatial + 2:
            raise ValueError(
                f"UNet config.sample_size must be spatial+2 (replicate pad by 1 on each side); "
                f"expected {spatial + 2}, got {inner}"
            )
        self.unet = unet
        self.spatial = spatial
        self.inner_spatial = inner
        self.pad_mode = pad_mode

    @property
    def dtype(self) -> torch.dtype:
        """Proxy inner UNet dtype; required by diffusers pipeline_utils.to()."""
        return self.unet.dtype

    @property
    def device(self) -> torch.device:
        """Proxy inner UNet device; required by DiffusionPipeline.device."""
        return next(self.parameters()).device

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        h, w = sample.shape[-2:]
        pad_crop = h == self.spatial and w == self.spatial

        if pad_crop:
            x = F.pad(sample, (1, 1, 1, 1), mode=self.pad_mode)
        elif h == self.inner_spatial and w == self.inner_spatial:
            x = sample
        else:
            raise ValueError(
                f"Expected spatial size {self.spatial}×{self.spatial} or {self.inner_spatial}×{self.inner_spatial}, "
                f"got {h}×{w}."
            )

        out = self.unet(x, timestep, class_labels=class_labels, return_dict=True)
        pred = out.sample

        if pad_crop and pred.shape[-2:] == (self.inner_spatial, self.inner_spatial):
            pred = pred[..., 1:-1, 1:-1]

        if not return_dict:
            return (pred,)
        return UNet2DOutput(sample=pred)


def save_openfwi_checkpoint(wrapper: OpenFWIUNetWrapper, path: Union[str, Path]) -> None:
    """Save ``OpenFWIUNetWrapper`` with ``torch.save`` (state dict + metadata to rebuild the module)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uc = wrapper.unet.config
    unet_cfg = uc.to_dict() if hasattr(uc, "to_dict") else dict(uc)
    torch.save(
        {
            "state_dict": wrapper.state_dict(),
            "spatial": wrapper.spatial,
            "pad_mode": wrapper.pad_mode,
            "unet_config": unet_cfg,
        },
        path,
    )


def load_openfwi_checkpoint(
    path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
) -> OpenFWIUNetWrapper:
    """Load ``OpenFWIUNetWrapper`` from ``save_openfwi_checkpoint`` output."""
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    unet = UNet2DModel.from_config(ckpt["unet_config"])
    wrapper = OpenFWIUNetWrapper(
        unet,
        spatial=int(ckpt["spatial"]),
        pad_mode=str(ckpt.get("pad_mode", "replicate")),
    )
    wrapper.load_state_dict(ckpt["state_dict"], strict=True)
    if torch_dtype is not None:
        wrapper = wrapper.to(dtype=torch_dtype)
    return wrapper


def resolve_pretrained_model_pt(pretrained: Path) -> Optional[Path]:
    """``path/to/model.pt`` or directory containing ``model.pt``."""
    pretrained = pretrained.resolve()
    if pretrained.is_file() and pretrained.suffix.lower() in (".pt", ".pth"):
        return pretrained
    cand = pretrained / "model.pt"
    if cand.is_file():
        return cand
    return None
