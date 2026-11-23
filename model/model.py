import re

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from base import BaseModel


# --------------------------------------------------------------------------- #
# U-Net from scratch
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, 2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNetScratch(BaseModel):
    def __init__(self, in_channels=3, num_classes=1, base_c=64, bilinear=False):
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, (base_c * 16) // factor)
        self.up1 = Up(base_c * 16, (base_c * 8) // factor, bilinear)
        self.up2 = Up(base_c * 8, (base_c * 4) // factor, bilinear)
        self.up3 = Up(base_c * 4, (base_c * 2) // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.outc = nn.Conv2d(base_c, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# --------------------------------------------------------------------------- #
# U-Net with a pretrained ResNet encoder
# --------------------------------------------------------------------------- #
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        total_in = in_channels // 2 + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetResNet(BaseModel):
    """U-Net with a torchvision ResNet34/50 encoder."""

    def __init__(
        self, backbone="resnet34", pretrained=True, in_channels=3, num_classes=1
    ):
        super().__init__()
        if backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            filters = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}' (use resnet34 or resnet50)"
            )

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, 7, stride=2, padding=3, bias=False
            )

        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder_maxpool = resnet.maxpool
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4

        self.dec4 = DecoderBlock(filters[4], filters[3], 256)
        self.dec3 = DecoderBlock(256, filters[2], 128)
        self.dec2 = DecoderBlock(128, filters[1], 64)
        self.dec1 = DecoderBlock(64, filters[0], 32)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        s1 = self.encoder_conv1(x)
        s2 = self.layer1(self.encoder_maxpool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        d4 = self.dec4(s5, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.final_up(d1)


class UNetPretrained(BaseModel):
    """Uses segmentation_models_pytorch when installed, else the torchvision U-Net above."""

    def __init__(
        self, backbone="resnet34", pretrained=True, in_channels=3, num_classes=1
    ):
        super().__init__()
        self.use_smp = False
        try:
            import segmentation_models_pytorch as smp

            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=in_channels,
                classes=num_classes,
            )
            self.use_smp = True
        except ImportError:
            self.model = UNetResNet(backbone, pretrained, in_channels, num_classes)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------------- #
# SegFormer (Transformer encoder + all-MLP decoder), via Hugging Face transformers
# --------------------------------------------------------------------------- #
# MiT encoder presets (same as the official nvidia/mit-b0..b5 configs). Used to
# build the architecture offline, e.g. when loading one of our own checkpoints.
_SEGFORMER_COMMON = {
    "num_encoder_blocks": 4,
    "sr_ratios": [8, 4, 2, 1],
    "patch_sizes": [7, 3, 3, 3],
    "strides": [4, 2, 2, 2],
    "num_attention_heads": [1, 2, 5, 8],
    "mlp_ratios": [4, 4, 4, 4],
}
SEGFORMER_PRESETS = {
    "b0": {
        "hidden_sizes": [32, 64, 160, 256], "depths": [2, 2, 2, 2], "decoder_hidden_size": 256
    },
    "b1": {
        "hidden_sizes": [64, 128, 320, 512], "depths": [2, 2, 2, 2], "decoder_hidden_size": 256
    },
    "b2": {
        "hidden_sizes": [64, 128, 320, 512], "depths": [3, 4, 6, 3], "decoder_hidden_size": 768
    },
    "b3": {
        "hidden_sizes": [64, 128, 320, 512], "depths": [3, 4, 18, 3], "decoder_hidden_size": 768
    },
    "b4": {
        "hidden_sizes": [64, 128, 320, 512], "depths": [3, 8, 27, 3], "decoder_hidden_size": 768
    },
    "b5": {
        "hidden_sizes": [64, 128, 320, 512], "depths": [3, 6, 40, 3], "decoder_hidden_size": 768
    },
}


def _segformer_variant(name):
    match = re.search(r"(?:^|[-/])b([0-5])(?:-|$)", name)
    if match is None:
        raise ValueError(
            f"Cannot infer the SegFormer variant (b0..b5) from '{name}'. "
            "Use e.g. 'nvidia/mit-b0' or 'nvidia/mit-b2'."
        )
    return "b" + match.group(1)


class SegFormer(BaseModel):
    """
    SegFormer for binary segmentation. The HF model predicts at 1/4 resolution, so the
    logits are bilinearly upsampled back to the input size.

    pretrained=True  -> ImageNet-pretrained MiT encoder from the Hugging Face hub
                        (the decoder head is freshly initialised).
    pretrained=False -> random init from the offline preset table (no network needed).
    """

    def __init__(
        self, name="nvidia/mit-b0", pretrained=True, in_channels=3, num_classes=1
    ):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SegFormer needs the 'transformers' package: pip install transformers "
                "(or: uv sync --extra transformers)"
            ) from e

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                name,
                num_labels=num_classes,
                num_channels=in_channels,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig(
                num_labels=num_classes,
                num_channels=in_channels,
                **_SEGFORMER_COMMON,
                **SEGFORMER_PRESETS[_segformer_variant(name)],
            )
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits  # [B, C, H/4, W/4]
        return F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )


def build_model(
    model_type="scratch",
    backbone="resnet34",
    pretrained=True,
    in_channels=3,
    num_classes=1,
    base_c=64,
    bilinear=False,
    segformer_name="nvidia/mit-b0",
):
    """Factory driven by the ``arch`` section of config.json."""
    kind = model_type.lower()
    if kind == "scratch":
        return UNetScratch(in_channels, num_classes, base_c, bilinear)
    if kind in ("pretrained", "resnet", "unet_pretrained"):
        return UNetPretrained(backbone, pretrained, in_channels, num_classes)
    if kind == "segformer":
        return SegFormer(segformer_name, pretrained, in_channels, num_classes)
    raise ValueError(
        f"Unknown model_type '{model_type}' (use 'scratch', 'pretrained' or 'segformer')"
    )


def load_model(checkpoint_path, device):
    """Rebuild a model from the config stored in a checkpoint and load its weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    arch = dict(config["arch"])
    arch["pretrained"] = False  # weights come from the checkpoint
    model = build_model(**arch).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config
