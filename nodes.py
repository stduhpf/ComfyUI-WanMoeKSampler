import torch
import numpy as np

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_sampling
from comfy.model_sampling import ModelSamplingDiscreteFlow, CONST

import latent_preview


def wan_ksampler(
    model_high_noise,
    model_low_noise,
    seed,
    steps,
    cfgs,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    boundary=0.875,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
    cfg_fall_ratio_high=0.5,
    cfg_fall_ratio_low=0.5,
):
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)

    # --- Determine switching point (boundary) ---
    sampling = model_high_noise.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
    timesteps = [sampling.timestep(sigma) / 1000 for sigma in sigmas.tolist()]
    split_at_step = steps
    for i, t in enumerate(timesteps):
        if i == 0:
            continue
        if t < boundary:
            split_at_step = i
            break

    print(f"Switching model at step {split_at_step}. High-noise runs {split_at_step} steps, Low-noise runs {steps - split_at_step} steps.")

    # Clamp user-defined start/end
    start_at = 0 if start_step is None else start_step
    end_at = steps if last_step is None else min(steps, last_step)
    high_noise_end_step = min(end_at, split_at_step)
    low_noise_start_step = max(start_at, split_at_step)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # --- Build configurable CFG schedules ---
    def build_cfg_schedule(start_cfg, total_steps, ratio):
        if total_steps <= 1:
            return np.array([start_cfg])
        fall_steps = int(total_steps * ratio)
        if fall_steps < 1:
            return np.ones(total_steps) * 1.0
        decay = np.linspace(start_cfg, 1.0, fall_steps)
        sustain = np.ones(total_steps - fall_steps)
        return np.concatenate([decay, sustain])

    cfg_high_schedule = build_cfg_schedule(cfgs[0], high_noise_end_step - start_at, cfg_fall_ratio_high)
    cfg_low_schedule = build_cfg_schedule(cfgs[1], end_at - low_noise_start_step, cfg_fall_ratio_low)

    # --- HIGH NOISE MODEL ---
    if start_at < high_noise_end_step:
        print(f"Running high noise model for steps {start_at}–{high_noise_end_step - 1} (CFG {cfgs[0]}→1 over {cfg_fall_ratio_high*100:.0f}% of steps)")
        callback = latent_preview.prepare_callback(model_high_noise, steps)
        latent_image = comfy.sample.fix_empty_latent_channels(model_high_noise, latent_image)
        current_latent = latent_image
        for i, cfg_val in enumerate(cfg_high_schedule, start=start_at):
            current_latent = comfy.sample.sample(
                model_high_noise,
                noise,
                steps,
                float(cfg_val),
                sampler_name,
                scheduler,
                positive,
                negative,
                current_latent,
                denoise=denoise,
                disable_noise=(low_noise_start_step < end_at) or disable_noise,
                start_step=i,
                last_step=i + 1,
                force_full_denoise=(low_noise_start_step >= end_at) or force_full_denoise,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        latent_image = current_latent

    # --- LOW NOISE MODEL ---
    if low_noise_start_step < end_at:
        print(f"Running low noise model for steps {low_noise_start_step}–{end_at - 1} (CFG {cfgs[1]}→1 over {cfg_fall_ratio_low*100:.0f}% of steps)")
        callback = latent_preview.prepare_callback(model_low_noise, steps)
        latent_image = comfy.sample.fix_empty_latent_channels(model_low_noise, latent_image)
        current_latent = latent_image
        for i, cfg_val in enumerate(cfg_low_schedule, start=low_noise_start_step):
            current_latent = comfy.sample.sample(
                model_low_noise,
                noise,
                steps,
                float(cfg_val),
                sampler_name,
                scheduler,
                positive,
                negative,
                current_latent,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=i,
                last_step=i + 1,
                force_full_denoise=force_full_denoise,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=seed,
            )
        latent_image = current_latent

    out = latent.copy()
    out["samples"] = latent_image
    return (out,)


# --- Helper for sigma shift ---
def set_shift(model, sigma_shift):
    model_sampling = model.get_model_object("model_sampling")
    if not model_sampling:
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced()

    model_sampling.set_parameters(shift=sigma_shift, multiplier=1000)
    model.add_object_patch("model_sampling", model_sampling)
    return model


# --- Simple Node ---
class WanMoeKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_high_noise": ("MODEL", {"tooltip": "High-noise expert model used for the early denoising phase."}),
                "model_low_noise": ("MODEL", {"tooltip": "Low-noise expert model used for the later denoising phase."}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Timestep boundary where the sampler switches from high-noise to low-noise model."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for noise generation; controls reproducibility."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Total number of denoising steps to perform."}),
                "cfg_high_noise": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Initial CFG (Classifier-Free Guidance) scale for the high-noise model."}),
                "cfg_low_noise": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Initial CFG scale for the low-noise model."}),
                "cfg_fall_ratio_high": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of high-noise model steps during which CFG linearly falls from start value to 1.0."}),
                "cfg_fall_ratio_low": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of low-noise model steps during which CFG linearly falls from start value to 1.0."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm used during denoising."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise schedule controlling how noise is removed per step."}),
                "sigma_shift": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Sigma shift factor that modifies the noise distribution for both models."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning (what you want to see)."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning (what you want to avoid)."}),
                "latent_image": ("LATENT", {"tooltip": "Input latent tensor to be denoised."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength; lower values retain more of the original latent structure."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Dual-model sampler with independent dynamic CFG decay ratios for each model."

    def sample(self, model_high_noise, model_low_noise, boundary, seed, steps, cfg_high_noise, cfg_low_noise, cfg_fall_ratio_high, cfg_fall_ratio_low, sampler_name, scheduler, sigma_shift, positive, negative, latent_image, denoise=1.0):
        model_high_noise = set_shift(model_high_noise, sigma_shift)
        model_low_noise = set_shift(model_low_noise, sigma_shift)
        return wan_ksampler(
            model_high_noise,
            model_low_noise,
            seed,
            steps,
            (cfg_high_noise, cfg_low_noise),
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            boundary=boundary,
            denoise=denoise,
            cfg_fall_ratio_high=cfg_fall_ratio_high,
            cfg_fall_ratio_low=cfg_fall_ratio_low,
        )


# --- Advanced Node ---
class WanMoeKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_high_noise": ("MODEL", {"tooltip": "High-noise expert model used for early denoising."}),
                "model_low_noise": ("MODEL", {"tooltip": "Low-noise expert model used for later refinement."}),
                "boundary": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Boundary (t_moe) determining where to switch from high- to low-noise model."}),
                "add_noise": (["enable", "disable"], {"tooltip": "Enable or disable noise addition at the start of denoising."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for noise generation."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of total denoising steps."}),
                "cfg_high_noise": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Starting CFG scale for the high-noise model."}),
                "cfg_low_noise": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Starting CFG scale for the low-noise model."}),
                "cfg_fall_ratio_high": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of high-noise steps where CFG linearly decays to 1.0."}),
                "cfg_fall_ratio_low": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of low-noise steps where CFG linearly decays to 1.0."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Select the sampler algorithm (e.g., euler, dpmpp_2m, etc.)."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler type controlling noise sigma progression."}),
                "sigma_shift": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Shift applied to sigma schedule for adjusting denoising behavior."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive text conditioning for guiding image generation."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative text conditioning for avoiding unwanted elements."}),
                "latent_image": ("LATENT", {"tooltip": "Input latent image tensor."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Optional: start denoising from this step index."}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "tooltip": "Optional: stop denoising at this step index."}),
                "return_with_leftover_noise": (["disable", "enable"], {"tooltip": "If enabled, retains leftover noise in the output latent instead of full denoising."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Advanced version of the dual-model sampler with precise control over noise behavior and CFG decay."

    def sample(
        self,
        model_high_noise,
        model_low_noise,
        boundary,
        add_noise,
        noise_seed,
        steps,
        cfg_high_noise,
        cfg_low_noise,
        cfg_fall_ratio_high,
        cfg_fall_ratio_low,
        sampler_name,
        scheduler,
        sigma_shift,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        model_high_noise = set_shift(model_high_noise, sigma_shift)
        model_low_noise = set_shift(model_low_noise, sigma_shift)

        force_full_denoise = return_with_leftover_noise != "enable"
        disable_noise = add_noise == "disable"

        return wan_ksampler(
            model_high_noise,
            model_low_noise,
            noise_seed,
            steps,
            (cfg_high_noise, cfg_low_noise),
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            boundary=boundary,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
            cfg_fall_ratio_high=cfg_fall_ratio_high,
            cfg_fall_ratio_low=cfg_fall_ratio_low,
        )
