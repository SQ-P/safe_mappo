"""Frozen-model diagnostics; virtual copies only, no Actor installation or fitting."""
from copy import deepcopy

import numpy as np
import torch


def kinematic_value(action,positions,velocities,goals,*,dt,acceleration,max_speed,
                    start_max,start_mean,bottleneck):
    """Differentiable unprojected checkpoint goal potential, without Guard gradients.

    Integration/distance statistics follow the action dtype; scalar potential
    ratios use float64 like the environment's Python-float arithmetic.
    Max, component clipping and norm kinks retain PyTorch subgradient semantics.
    """
    p,v,g = (torch.as_tensor(x,device=action.device,dtype=action.dtype) for x in (positions,velocities,goals))
    future = p+torch.clamp(v+action*acceleration*dt,-max_speed,max_speed)*dt
    def potential(x):
        distances = torch.linalg.vector_norm(x-g,dim=-1)
        residual = bottleneck*distances.max().double()/max(float(start_max),1e-6)
        residual = residual+(1.-bottleneck)*distances.mean().double()/max(float(start_mean),1e-6)
        return torch.clamp(1.-residual,-1.,1.)
    return potential(future)-potential(p)


def action_gradient(mean,value):
    x = mean.detach().clone().requires_grad_(True)
    score = value(torch.tanh(x))
    grad, = torch.autograd.grad(score,x)
    if not torch.isfinite(grad).all() or not torch.isfinite(score):
        raise ValueError("Nonfinite diagnostic action derivative")
    return grad.detach(),float(score.detach())


def cosine(a,b):
    norm = a.norm()*b.norm()
    return float((a*b).sum()/norm) if float(norm)>0 else None


def conflict_summary(vectors,layouts):
    """No zero-gradient pair is silently counted as agreement."""
    vectors = torch.stack(vectors)
    def describe(values):
        pairs = [cosine(values[i],values[j]) for i in range(len(values)) for j in range(i)]
        valid = [p for p in pairs if p is not None]
        norm_sum = values.norm(dim=1).sum()
        return {"vectors":len(values),"zero_vectors":int((values.norm(dim=1)==0).sum()),
            "pair_cosines":pairs,"defined_pairs":len(valid),"negative_pairs":sum(p<0 for p in valid),
            "cancellation_ratio":float(values.sum(0).norm()/norm_sum) if float(norm_sum)>0 else None}
    unique = sorted(set(layouts))
    grouped = torch.stack([vectors[[i for i,g in enumerate(layouts) if g == layout]].mean(0) for layout in unique])
    return {"per_state":describe(vectors),"per_layout":describe(grouped),"layout_order":unique,
        "note":"Descriptive derivative geometry, not causal identification or efficacy"}


def layout_weights(layouts,device):
    counts = {g:layouts.count(g) for g in set(layouts)}
    weights = torch.tensor([1./counts[g] for g in layouts],device=device)
    return weights/weights.sum()


def virtual_adapter(adapter,direction,step):
    if step<0 or not np.isfinite(step) or not torch.isfinite(direction).all():
        raise ValueError("Invalid virtual parameter change")
    candidate = deepcopy(adapter)
    offset = 0
    with torch.no_grad():
        for p in candidate.parameters():
            size = p.numel()
            p.add_(float(step)*direction[offset:offset+size].reshape_as(p))
            offset += size
    if offset != direction.numel():
        raise ValueError("Parameter direction size differs")
    return candidate


def residual_rms(adapter,features,weights):
    with torch.no_grad():
        residual = adapter(features)
        return float(torch.sqrt((weights*residual.flatten(1).square().sum(1)).sum()))


def match_calibration_budget(adapter,direction,features,weights,target):
    """One scalar for all states, using only caller-supplied Calibration features.

    At zero initialization only the up layer has a nonzero Jacobian. Along
    that direction, residual magnitude is monotone; never solve against Shadow
    gains or per-state outcomes. Cap and iteration counts are numerical limits.
    """
    if target<=0 or not np.isfinite(target):
        raise ValueError("Invalid pre-tanh RMS target")
    if float(direction.norm()) == 0:
        return 0.,{"status":"ZERO_DIRECTION","target_rms":target,"achieved_rms":0.}
    offset = 0
    for name,p in adapter.named_parameters():
        part = direction[offset:offset+p.numel()]
        if not name.startswith("up.") and bool((part != 0).any()):
            raise ValueError("Budget solver requires zero-initialized up-only tangent")
        offset += p.numel()
    if adapter.up.weight.detach().abs().max().item() or adapter.up.bias.detach().abs().max().item():
        raise ValueError("Budget solver requires zero output initialization")
    def measure(step):
        return residual_rms(virtual_adapter(adapter,direction,step),features,weights)
    low,high = 0.,.01
    while high<1024. and measure(high)<target:
        high = min(2.*high,1024.)
    if measure(high)<target:
        return None,{"status":"UNATTAINABLE_WITH_NUMERICAL_CAP","target_rms":target,"achieved_rms":measure(high)}
    for _ in range(40):
        middle = (low+high)/2.
        if measure(middle)<target:
            low = middle
        else:
            high = middle
    return high,{"status":"MATCHED_ON_CALIBRATION","target_rms":target,"achieved_rms":measure(high),
        "parameter_step":high,"scope":"single scalar, layout-weighted Calibration joint-action RMS"}


def tangent_trace(adapter,features,mean,value,jacobian):
    grad,score = action_gradient(mean,value)
    pullback = jacobian.T@grad.reshape(-1)
    # Independently verify the chain rule through the actual shared module.
    q = value(torch.tanh(mean.detach()+adapter(features.detach())))
    direct = torch.autograd.grad(q,tuple(adapter.parameters()),allow_unused=True)
    direct = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(adapter.parameters(),direct)]).detach()
    if not torch.allclose(pullback,direct,atol=1e-8,rtol=2e-5):
        raise RuntimeError("Action-to-parameter chain-rule mismatch")
    offset,layers = 0,{}
    for name,p in adapter.named_parameters():
        size = p.numel()
        layers[name] = {"parameters":size,"gradient_norm":float(pullback[offset:offset+size].norm()),
            "nonzero_jacobian_columns":int((jacobian[:,offset:offset+size]!=0).any(0).sum())}
        offset += size
    return {"gradient":grad,"parameter_gradient":pullback.detach(),"score":score,"layers":layers,
        "chain_rule_max_error":float((pullback-direct).abs().max())}
