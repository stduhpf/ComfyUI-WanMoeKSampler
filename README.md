This repo was forked, due to the original not being maintained since it's creation, 2 months ago.

# KSampler for Wan 2.2 MoE for ComfyUI

These nodes are made to support "Mixture of Expert" Flow models with the architecture of Wan2.2 A14B (With a high noise expert and low noise expert).
Instead of guessing the denoising step at which to swap from tyhe high noise model to the low noise model, this node automatically chanage to the low noise model when we reach the diffusion timestep at which the signal to noise ratio is supposed to be 1:1.


## Installation

To install this node, follow these steps:

1. Clone this repository into your ComfyUI custom nodes directory.
2. Restart ComfyUI to load the new node.

```bash
git clone https://github.com/stduhpf/ComfyUI--WanMoeKSampler.git /path/to/ComfyUI/custom_nodes/WanMoeKSampler
```

## Usage

See workflows included in this repository for basic usage.

### About the `boundary` parameter:

This correspond to the diffusion timestep around which the model used is supposed to start using the low noise expert. For Wan 2.2 T2V, this value should be `0.875`,  For Wan 2.2 I2V, the value should be `0.900`. Using other values might still work.

Note that diffusion timesteps is NOT the same thing as denoising steps at all. You could think of the diffusion timesetp roughly as how much noise is added in the image (during training). At timestep `0`, the image is clean, with no noise added.  At a timestep of `1`, the image/video is pure noise. And for Wan2.2 a14B T2V model, around timestep `0.875`(`0.9` for I2V), the video should be half noise, half useful data. The timestep is realated to the corresponding denoising step with a non-linear relationship that depends on the total number of steps, the sampling method used, and the noise scheduler (and sigma shift).


## License

This project mostly contains code copy-pasted from ComfyUI, which is licenced under GPL3.0. Therefore it is also licenced under GPL 3.0. (see LICENCE file for more details)
