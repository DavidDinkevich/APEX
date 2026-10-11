# Third-party code

This repository vendors and patches public codebases. Their licenses remain in the vendored trees.

- **DC-AE trainer** (`vae/`): [lavinal712/AutoencoderKL](https://github.com/lavinal712/AutoencoderKL) branch `dc-ae` (MIT; includes Stability AI SGM code). Architecture from [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) DC-AE.
- **SiT** (`sit/`): [willisma/SiT](https://github.com/willisma/SiT) (MIT).
- **Latent precomputation** (`sit/extract_features.py`, `sit/datasets/img_latent_dataset.py`, `sit/run_extraction.sh`, and the image transform in `sit/tokenizer/dcae.py`): [hustvl/LightningDiT](https://github.com/hustvl/LightningDiT) (MIT, Copyright 2024 HUST Vision Lab).
