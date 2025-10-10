from .nodes import (
    WanMoeKSampler,
    WanMoeKSamplerAdvanced,
)

NODE_CLASS_MAPPINGS = {
    "WanMoeKSampler": WanMoeKSampler,
    "WanMoeKSamplerAdvanced": WanMoeKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanMoeKSampler": "Wan MoE KSampler",
    "WanMoeKSamplerAdvanced": "Wan MoE KSampler (Advanced)",
}
