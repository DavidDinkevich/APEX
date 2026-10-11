# Image transform copied from LightningDiT tokenizer/vavae.py
# https://github.com/hustvl/LightningDiT
# MIT License, Copyright (c) 2024 HUST Vision Lab

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

VAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vae"))
sys.path.insert(0, VAE_ROOT)

from sgm.util import instantiate_from_config


class DC_AE:
    def __init__(self, config, img_size=512):
        self.config = OmegaConf.load(config)
        self.ckpt_path = self.config.ckpt_path
        self.img_size = img_size
        self.load()

    def load(self):
        self.model = instantiate_from_config(self.config.model)
        self.model.apply_ckpt(self.ckpt_path)
        self.model.cuda().eval()
        return self

    def img_transform(self, p_hflip=0, img_size=None):
        """Image preprocessing transforms
        Args:
            p_hflip: Probability of horizontal flip
            img_size: Target image size, use default if None
        Returns:
            transforms.Compose: Image transform pipeline
        """
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images):
        with torch.no_grad():
            return self.model.encode(images.cuda())


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
