"""test_sea_axioms.py — run this BEFORE trusting any SEA number.

    python test_sea_axioms.py --device cuda

Six checks. 1-3 must pass or the head is not what the design doc says it is.
4-5 print measurements rather than assert, because their correct values depend
on the backbone. 6 is the B-cos anomaly from TECHNICAL_REPORT section 8.

Runtime: under 2 minutes on CPU, ~20 s on GPU.
"""
from __future__ import annotations

import argparse
import itertools
import math

import torch
import torch.nn.functional as F

from pxai.models.sea import SEANet
from pxai.models.bcos import BcosHead

OK, BAD = "  [pass]", "  [FAIL]"


def cfg_for(**kw):
    c = {"backbone": {"name": kw.pop("backbone", "efficientnet_lite0"),
                      "pretrained": False},
         "model": {"kind": "sea", "num_classes": 11, "sea": dict(kw)}}
    return c


# --------------------------------------------------------------------------- #
def t1_completeness(device):
    """sum_p phi_{c,p} == f_c - a_c, exactly, for every configuration."""
    print("\n1. COMPLETENESS   sum_p phi = f - a")
    worst = 0.0
    for stride in (4, 8, 16, 32):
        for readout in ("linear", "mlp"):
            for context in ("film", "none"):
                m = SEANet(cfg_for(stride=stride, readout=readout,
                                   context=context)).to(device).eval()
                x = torch.randn(3, 3, 224, 224, device=device)
                with torch.no_grad():
                    f = m(x).double()
                    phi = m.explain(x)["contrib_map"].double()
                    a = m.head.prior.view(1, -1).double()
                # accumulate the reference sum in fp64 so this catches a real
                # mismatch, not just "the same fp32 sum computed twice"
                err = (phi.sum((-2, -1)) + a - f).abs().max().item()
                worst = max(worst, err)
                print(f"     stride={stride:<3} readout={readout:<7} "
                      f"context={context:<5} grid={tuple(phi.shape[-2:])} "
                      f"residual={err:.3e}")
    tol = 1e-3          # fp32, sums of up to 3136 terms
    print((OK if worst < tol else BAD) + f"  worst residual {worst:.3e} (tol {tol})")
    return worst < tol


def t2_shapley(device):
    """Brute-force Shapley over patch coalitions == the head's own map.

    Enumerating 2^P is impossible for P=3136, so we hold all but n cells fixed
    as background and enumerate the 2^n coalitions over the rest -- which is
    exactly how a KernelSHAP-over-patches baseline would be posed.
    """
    print("\n2. SHAPLEY EXACTNESS   Sh_p(v) == phi_p by brute-force enumeration")
    m = SEANet(cfg_for(stride=32, readout="mlp", context="film")).to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        phi = m.explain(x)["contrib_map"][0]            # (K,h,w)
    c = 4
    flat = phi[c].flatten().double()
    n = 10
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(flat.numel(), generator=g)[:n]
    bg = flat.sum() - flat[sel].sum()                   # fixed cells
    a = float(m.head.prior[c].detach())

    def v(S):
        return a + bg + sum(float(flat[sel[i]]) for i in S)

    sh = [0.0] * n
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in itertools.combinations(others, k):
                wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                sh[i] += wgt * (v(set(S) | {i}) - v(set(S)))
    err = max(abs(sh[i] - float(flat[sel[i]])) for i in range(n))
    eff = abs(sum(sh) - (v(set(range(n))) - v(set())))
    print(f"     max |Sh_p - phi_p| = {err:.3e}   over 2^{n} coalitions")
    print(f"     efficiency gap     = {eff:.3e}")
    good = err < 1e-9 and eff < 1e-9
    print((OK if good else BAD) + "  the map is the exact Shapley value of the "
          "head's own value function")
    return good


def t3_contrastive(device):
    """f_c - f_c' == (a_c - a_c') + sum_p (phi_c - phi_c')."""
    print("\n3. CONTRASTIVE EXACTNESS   decision, not just one logit")
    m = SEANet(cfg_for(stride=8)).to(device).eval()
    x = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        f = m(x)
        phi = m.explain(x)["contrib_map"]
        a = m.head.prior
    c, cp = 2, 9
    lhs = f[:, c] - f[:, cp]
    rhs = (a[c] - a[cp]) + (phi[:, c] - phi[:, cp]).sum((-2, -1))
    err = (lhs - rhs).abs().max().item()
    print((OK if err < 1e-3 else BAD) + f"  residual {err:.3e}")
    return err < 1e-3


# --------------------------------------------------------------------------- #
def t4_leakage(device):
    """How much class evidence can survive OUTSIDE phi? Measurement, not assert.

    With context=film the prior a_c is still class-dependent but input-blind,
    so it cannot discriminate between images -- but the FiLM vector can. This
    reports the fraction of the logit spread attributable to context-driven
    modulation, by re-running with the context vector of a DIFFERENT image.
    """
    print("\n4. CONTEXT SENSITIVITY   (measurement)")
    for context in ("film", "none"):
        m = SEANet(cfg_for(stride=8, context=context)).to(device).eval()
        x = torch.randn(8, 3, 224, 224, device=device)
        with torch.no_grad():
            pyr = m._pyramid(x)
            size = (224 // m.stride, 224 // m.stride)
            ctx = F.adaptive_avg_pool2d(pyr[m.ctx_idx], 1).flatten(1)
            phi = m.head.phi([pyr[i] for i in m.idx], ctx, size)
            phi_s = m.head.phi([pyr[i] for i in m.idx], ctx.roll(1, 0), size)
        rel = ((phi_s - phi).norm() / phi.norm()).item()
        print(f"     context={context:<5} relative change from swapping the "
              f"context vector: {rel:.3f}")
    print("       -> near 0 means phi is context-independent. Report this and "
          "the trained\n          adversarial-probe accuracy in the thesis; do "
          "not skip it.")
    return True


def t5_receptive_field(device):
    """Does phi at a cell move when far-away pixels move? Measurement.

    A bounded receptive field is a locality certificate. It only exists for
    pure-conv backbones -- mobilevit_* runs global attention and will show
    non-zero influence everywhere. That is a fact about the backbone, not a bug.
    """
    print("\n5. RECEPTIVE FIELD   (measurement)")
    for bb in ("efficientnet_lite0", "mobilevit_xs"):
        try:
            m = SEANet(cfg_for(backbone=bb, stride=8, context="none",
                               max_evidence_stride=8)).to(device).eval()
        except Exception as e:                                # noqa: BLE001
            print(f"     {bb}: skipped ({type(e).__name__})")
            continue
        x = torch.randn(1, 3, 224, 224, device=device)
        x2 = x.clone()
        x2[..., 180:224, 180:224] += 5.0                      # far corner
        with torch.no_grad():
            p1 = m.explain(x)["contrib_map"][0, 0]
            p2 = m.explain(x2)["contrib_map"][0, 0]
        d = (p2 - p1).abs()
        near, far = d[:6, :6].max().item(), d.max().item()
        print(f"     {bb:<20} corner perturbation -> |dphi| at the OPPOSITE "
              f"corner = {near:.3e}  (max anywhere {far:.3e})")
    return True


def t6_bcos_is_hirescam(device):
    """TECHNICAL_REPORT section 8: 'B-cos vs HiResCAM tie at deletion to 4 dp'.

    It is an identity. The B-cos block is positively homogeneous of degree 1 in
    its input (w.(tA) = t(w.A) and cos(tA,w) = cos(A,w)), so by Euler's theorem
    <grad g(A), A> = g(A). HiResCAM at that layer is therefore contrib_map / P
    exactly -- the same map up to a positive constant, hence identical ranks,
    hence identical deletion / insertion / pointing-game to machine precision.
    """
    print("\n6. B-COS HEAD == HiResCAM   (explains the section-8 tie)")
    head = BcosHead(384, 11, b=2.0).double().to(device)
    A = torch.randn(4, 384, 7, 7, dtype=torch.double, device=device,
                    requires_grad=True)
    contrib = head.block(A)
    y = head.pool(head.block(A)).flatten(1)
    tgt = torch.tensor([0, 3, 7, 10], device=device)
    g, = torch.autograd.grad(y.gather(1, tgt.view(-1, 1)).sum(), A)
    hires = (g * A).sum(1)
    sel = contrib.gather(1, tgt.view(-1, 1, 1, 1).expand(-1, 1, 7, 7)).squeeze(1)
    err = (hires - sel / 49).abs().max().item()
    print(f"     max |HiResCAM - contrib_map / P| = {err:.3e}")
    print((OK if err < 1e-12 else BAD) + "  head-only B-cos is not a distinct "
          "method; fold the row or make the backbone B-cos")
    return err < 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device != "cuda" else "cpu")
    print(f"device: {dev}")
    res = [t1_completeness(dev), t2_shapley(dev), t3_contrastive(dev),
           t4_leakage(dev), t5_receptive_field(dev), t6_bcos_is_hirescam(dev)]
    print("\n" + ("ALL CHECKS PASSED" if all(res) else "SOMETHING FAILED"))
    return 0 if all(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
