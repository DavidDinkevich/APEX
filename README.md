# APEX

**APEX: Asynchronous Prefix Denoising for Extreme Compression**

David Dinkevich, Nisan Chiprut, Yoav HaCohen, Dani Lischinski  
Lightricks / Hebrew University of Jerusalem

Paper reference implementation. Train a DC-AE with level prefix supervision in `vae/`, then a SiT with a staggered clock in `sit/`.

The academic project page is on the `gh-pages` branch. GitHub Pages is **not** enabled; preview it locally with `python3 -m http.server` from that branch.

## Example: f64c8-32-128

Official `dc-ae-f64c128-in-1.0` latents (8×8×128) with channel prefixes 8 / 32 / 128.

1. **Autoencoder.** From `vae/`, run the three-phase Lightning configs `configs/apex-f64c8-32-128_phase{1,2,3}.yaml`. See `vae/README.md`. Each step samples a prefix, zeros later channels, and scores BOX-resized reconstructions with the stock DC-AE loss.

2. **Latents.** Encode ImageNet with the trained AE (`AutoencoderDC.encode`) to `(128, 8, 8)` float32 `.npy` files in ImageFolder class directories. Point SiT `--data-path` at that root.

3. **Diffusion.** From `sit/`:

```bash
LATENT_DIR=/path/to/latents NPROC=8 bash scripts/train_f64c8-32-128-p09-g025.sh
CKPT=/path/to/checkpoint.pt bash scripts/sample_f64c8-32-128-p09-g025.sh
```

SiT-XL/1, gap 0.25, `p_L0` 0.9, 250-step staggered Euler, CFG on all latent channels. Sampling writes latents (`sample_latents.pt`); decode with the same AE.

Install each tree from its own file: `vae/requirements.txt` and `sit/environment.yml`.
