"""Post-hoc baselines (the methods we benchmark AGAINST), via Captum.

Each returns a (B, 1, H, W) attribution in input space so the Quantus harness
can score them on the same footing as ante-hoc explanations. We also expose a
`cost` dict (forward/backward passes) so the cost axis is recorded per method,
not just wall-clock — this is the heart of the "lighter than SOTA" claim.

Every function explains the model it is HANDED. Do not close over an outer model:
Quantus's sanity_check passes randomised copies in, and a captured model makes the
explanation invariant to randomisation (a silent, maximal sanity-check failure).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from captum.attr import (LayerGradCam, IntegratedGradients,
                         KernelShap, Lime)

# LIME/KernelSHAP surrogate granularity. Captum's default is one feature per
# ELEMENT: 3*224*224 = 150,528 features fit from n_samples=1000 draws, per image.
# That regression is unidentifiable — the Lasso zeroes almost everything, giving
# constant attribution columns (the `invalid value encountered in divide` warnings
# out of np.corrcoef downstream). A 16x16 grid groups all channels of each 14x14
# patch into one feature: 256 features from 1000 samples, which is estimable.
SUPERPIXEL_GRID = 16

# Captum evaluates this many perturbations per forward pass. Higher = faster and
# more VRAM (effective batch = input batch x this). Drop to 1 on CUDA OOM.
PERTURBATIONS_PER_EVAL = 4


def _to_input_space(attr, size):
    a = attr.sum(1, keepdim=True) if attr.shape[1] > 1 else attr
    a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    return a


def _superpixel_mask(x, grid: int = SUPERPIXEL_GRID):
    """(1,C,H,W) long tensor of group ids; all channels of a patch share an id.

    Nearest-neighbour tiling done in integer arithmetic so it is exact for sizes
    that do not divide evenly (e.g. 224/16 = 14 exactly, but 227/16 would not).
    """
    _, C, H, W = x.shape
    gh, gw = min(grid, H), min(grid, W)
    ids = torch.arange(gh * gw, device=x.device, dtype=torch.long).view(gh, gw)
    row = (torch.arange(H, device=x.device) * gh // H).clamp(max=gh - 1)
    col = (torch.arange(W, device=x.device) * gw // W).clamp(max=gw - 1)
    mask = ids[row][:, col]                                  # (H, W)
    return mask.view(1, 1, H, W).expand(1, C, H, W).contiguous()


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
    attr = ig.attribute(x, target=target, n_steps=steps,
                        internal_batch_size=max(1, x.shape[0] // 4))
    return _to_input_space(attr.abs(), x.shape[-2:]), {"fwd": steps, "bwd": steps}


def lime(model, x, target, n_samples: int = 1000, grid: int = SUPERPIXEL_GRID):
    mask = _superpixel_mask(x, grid)
    lm = Lime(lambda t: model(t))
    attr = lm.attribute(x, target=target, n_samples=n_samples,
                        feature_mask=mask,
                        perturbations_per_eval=PERTURBATIONS_PER_EVAL)
    # n_samples forward passes regardless of the mask — the mask fixes the
    # surrogate fit, not the sampling cost. EXPLAINER_PASSES stays at 1000.
    return _to_input_space(attr.abs(), x.shape[-2:]), {
        "fwd": n_samples, "bwd": 0, "n_features": int(mask.max().item()) + 1}


def kernel_shap(model, x, target, n_samples: int = 1000, grid: int = SUPERPIXEL_GRID):
    mask = _superpixel_mask(x, grid)
    ks = KernelShap(lambda t: model(t))
    attr = ks.attribute(x, target=target, n_samples=n_samples,
                        feature_mask=mask,
                        perturbations_per_eval=PERTURBATIONS_PER_EVAL)
    return _to_input_space(attr.abs(), x.shape[-2:]), {
        "fwd": n_samples, "bwd": 0, "n_features": int(mask.max().item()) + 1}


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
        # last_conv_layer(model) resolves against the model passed in, so this
        # stays correct when Quantus hands us a randomised copy.
        return fn(model, x, target, kw.get("layer") or last_conv_layer(model))
    return fn(model, x, target)