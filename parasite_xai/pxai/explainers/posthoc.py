"""Post-hoc baselines (the methods we benchmark AGAINST), via Captum.

Each returns a (B, 1, H, W) attribution in input space so the Quantus harness
can score them on the same footing as ante-hoc explanations. We also expose a
`cost` dict (forward/backward passes) so the cost axis is recorded per method,
not just wall-clock — this is the heart of the "lighter than SOTA" claim.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from captum.attr import (LayerGradCam, IntegratedGradients,
                         KernelShap, Lime)


def _to_input_space(attr, size):
    a = attr.sum(1, keepdim=True) if attr.shape[1] > 1 else attr
    a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    return a


def grad_cam(model, x, target, layer):
    gc = LayerGradCam(lambda t: model(t), layer)
    attr = gc.attribute(x, target=target)
    return _to_input_space(attr, x.shape[-2:]), {"fwd": 1, "bwd": 1}


def hires_cam(model, x, target, layer):
    """HiResCAM = element-wise (not channel-averaged) grad*activation — Draelos & Carin 2020.
    Faithfulness guarantee Grad-CAM lacks; same ~1 backward-pass cost."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    out = model(x)
    model.zero_grad(set_to_none=True)
    out.gather(1, target.view(-1, 1)).sum().backward()
    h1.remove(); h2.remove()
    cam = (acts["a"] * grads["g"]).sum(1, keepdim=True).clamp(min=0)
    return _to_input_space(cam, x.shape[-2:]), {"fwd": 1, "bwd": 1}


def integrated_gradients(model, x, target, steps: int = 32):
    ig = IntegratedGradients(lambda t: model(t))
    attr = ig.attribute(x, target=target, n_steps=steps)
    return _to_input_space(attr.abs(), x.shape[-2:]), {"fwd": steps, "bwd": steps}


def lime(model, x, target, n_samples: int = 1000):
    lm = Lime(lambda t: model(t))
    attr = lm.attribute(x, target=target, n_samples=n_samples)
    return _to_input_space(attr.abs(), x.shape[-2:]), {"fwd": n_samples, "bwd": 0}


def kernel_shap(model, x, target, n_samples: int = 1000):
    ks = KernelShap(lambda t: model(t))
    attr = ks.attribute(x, target=target, n_samples=n_samples)
    return _to_input_space(attr.abs(), x.shape[-2:]), {"fwd": n_samples, "bwd": 0}


def last_conv_layer(model):
    """Best-effort: return the last Conv2d module for CAM methods."""
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


REGISTRY = {
    "gradcam": grad_cam,
    "hirescam": hires_cam,
    "integrated_gradients": integrated_gradients,
    "lime": lime,
    "kernelshap": kernel_shap,
}


def explain_posthoc(name, model, x, target, **kw):
    fn = REGISTRY[name]
    if name in ("gradcam", "hirescam"):
        return fn(model, x, target, kw.get("layer") or last_conv_layer(model))
    return fn(model, x, target)
