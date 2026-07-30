"""Cost axis (protocol §4c) — the lightweight claim, measured not asserted.

Records: parameter count, model size (MB), FLOPs, and CPU latency (an edge proxy
for Raspberry-Pi / mid-range Android). Also reports passes-per-explanation for
each explainer so the ">=100x cheaper than KernelSHAP" claim is auditable.
"""
from __future__ import annotations

import time
import torch


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    b = sum(p.numel() * p.element_size() for p in model.parameters())
    b += sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return b / 1e6


def flops(model, img_size: int = 224) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        x = torch.randn(1, 3, img_size, img_size, device=device)
        return float(FlopCountAnalysis(model.eval(), x).total())
    except Exception as e:
        print(f"[cost] FLOPs unavailable: {e}")
        return None


@torch.no_grad()
def latency_cpu(model, img_size: int = 224, iters: int = 50, warmup: int = 5) -> float:
    """Median forward latency in ms on CPU (edge proxy).

    NOTE: model.cpu() is in-place, so without saving+restoring the original
    device this would permanently strand the shared model object on CPU,
    breaking every GPU call that runs after cost_report() (e.g. faithfulness
    eval). We restore the model's original device/mode afterward.
    """
    orig_device = next(model.parameters()).device
    was_training = model.training
    try:
        model.eval().cpu()
        x = torch.randn(1, 3, img_size, img_size)
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return ts[len(ts) // 2]
    finally:
        model.to(orig_device)
        model.train(was_training)


def cost_report(model, img_size: int = 224) -> dict:
    # Compute FLOPs ONCE. The old ternary called flops() in both the condition
    # and the value, tracing the model through fvcore twice — which is why the
    # "Unsupported operator aten::gelu" block appears twice in the logs. It also
    # mapped a legitimate 0.0 to None; `is not None` is the right test.
    f = flops(model, img_size)
    return {
        "params_M": round(count_params(model) / 1e6, 3),
        "size_MB": round(model_size_mb(model), 2),
        "flops_G": round(f / 1e9, 3) if f is not None else None,
        "latency_cpu_ms": round(latency_cpu(model, img_size), 2),
    }


# passes-per-explanation lookup for the cost-vs-faithfulness frontier
EXPLAINER_PASSES = {
    "gradcam": 2, "hirescam": 2, "integrated_gradients": 64,
    "lime": 1000, "kernelshap": 1000,
    "protopnet": 1, "cbm": 1, "bcos": 1, "amortized": 1,
}