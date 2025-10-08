# Modified from sam.build_sam.py
from typing import Dict, Any
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import torch.nn as nn


from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

from .messageTool import MessageTool


def _load_checkpoint_no_image_encoder(model, ckpt_path):
    if ckpt_path is None:
        return

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = raw.get("model", raw)

    filtered_sd = {
        k: v
        for k, v in sd.items()
        if "image_encoder" not in k.lower()
    }

    missing_keys, unexpected_keys = model.load_state_dict(filtered_sd, strict=False)
    missing = [k for k in missing_keys if "fakeneck" not in k.lower() and "fakeimageencoder" not in k.lower()]
    unexpected = [k for k in unexpected_keys if "fakeneck" not in k.lower() and "fakeimageencoder" not in k.lower()]

    if missing:
        MessageTool.MessageLog(f">>> Missing keys: {missing}")
    if unexpected:
        MessageTool.MessageLog(f">>> Unexpected keys: {unexpected}")

def build_sam2_no_encoder(
    config_file,
    ckpt_path=None,
    device="cuda",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint_no_image_encoder(model, ckpt_path)
    model = model.to(device)
    model.eval()
    
    return model


class FakeNeck(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self):
        # This method is not actually called by FakeImageEncoder
        pass


class FakeImageEncoder(nn.Module):
    """
    A mock ImageEncoder used when image features are precomputed and
    the actual encoder is not needed. It mimics the interface of a real encoder.
    """
    def __init__(self, neck: nn.Module, img_size: int = 1024) -> None:
        super().__init__()
        self.img_size = img_size
        self.neck = neck

    def forward(self, x: Any) -> Dict[str, Any]:
        """
        Mimics the output of a real ImageEncoder.
        Returns a dict with 'vision_features' as the input tensor.
        """
        return {
            "vision_features": x,
            "vision_pos_enc": None,
            "backbone_fpn": None,
        }


class Sam2PredictorNoImgEncoder(SAM2ImagePredictor):
    def __init__(
            self,
            sam_model: SAM2Base,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
    ) -> None:
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.mask_threshold = mask_threshold
        self.reset_predictor()

    def set_image_feature(self, sam2_features: dict, img_size: list[tuple[int, int]]=None):
        
        for k, v in sam2_features.items():
            if isinstance(v, list):
                sam2_features[k] = [torch.as_tensor(x, device=self.device) for x in v]
            else:
                sam2_features[k] = torch.as_tensor(v, device=self.device)
        self._features = sam2_features
        self._orig_hw = img_size
        self._is_image_set = True
