from .nodes import WanMoeKSampler,WanMoeKSamplerAdvanced,SplitSigmasAtT

NODE_CLASS_MAPPINGS = {
    "WanMoeKSampler":WanMoeKSampler,
    "WanMoeKSamplerAdvanced":WanMoeKSamplerAdvanced,
    "SplitSigmasAtT":SplitSigmasAtT
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanMoeKSampler": "Wan MoE KSampler",
    "WanMoeKSamplerAdvanced": "Wan MoE KSampler (Advanced)",
    "SplitSigmasAtT": "Split sigmas at timestep"
}