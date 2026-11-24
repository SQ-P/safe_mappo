"""Separate one-step executed-value model; never installed in the PPO trainer."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.algorithms.ppo.networks import SafetyAlignmentAdapter


class ExecutedValueCritic(nn.Module):
    def __init__(self,context_dim,agents=3,hidden_dim=128):
        super().__init__()
        self.context_dim,self.agents = int(context_dim),int(agents)
        self.network = nn.Sequential(nn.Linear(self.context_dim+2*self.agents,hidden_dim),nn.Tanh(),
                                     nn.Linear(hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))

    def forward(self,context,action):
        if action.shape[-2:] != (self.agents,2) or context.shape[-1] != self.context_dim:
            raise ValueError("PPCSA value input shape mismatch")
        return self.network(torch.cat([context,action.flatten(start_dim=-2)],dim=-1)).squeeze(-1)

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def executed_gap_loss(current_value,reference_value,weights,valid_reference,*,deadzone=0.):
    if deadzone < 0 or not np.isfinite(deadzone):
        raise ValueError("PPCSA deadzone must be finite and nonnegative")
    valid = valid_reference.detach().bool()
    if weights.shape != current_value.shape or valid.shape != current_value.shape or reference_value.shape != current_value.shape:
        raise ValueError("PPCSA gap shapes must match")
    if not torch.isfinite(weights).all() or bool((weights<0).any()):
        raise ValueError("PPCSA weights must be finite and nonnegative")
    if not torch.isfinite(current_value).all() or not torch.isfinite(reference_value[valid]).all():
        raise ValueError("PPCSA value evidence must be finite")
    weight = weights.detach().clamp(min=0.)*valid
    # Invalid teachers are never read into a positive signal (including NaN).
    ref = torch.where(valid,reference_value.detach(),current_value.detach())
    return (weight*torch.relu(ref-current_value-float(deadzone))).sum()/weight.sum().clamp(min=1.)


def zero_adapter(hidden_dim=128):
    # Reproducible diagnostic initialization without changing caller RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261121)
        return SafetyAlignmentAdapter(hidden_dim,2,rank=16,scale=.15)


def bounded_probe(mean,direction,epsilon):
    if not 0. < epsilon <= .15 or not torch.isfinite(direction).all():
        raise ValueError("probe must have finite direction and epsilon in (0,.15]")
    unit = direction/direction.norm().clamp(min=1e-12)
    delta = (float(epsilon)*unit).clamp(-.15,.15)
    return torch.tanh(mean+delta),delta


def adapter_jacobian(adapter,features):
    """Jacobian of the actual shared Adapter, not independent action targets."""
    params = tuple(adapter.parameters())
    outputs = adapter(features).reshape(-1)
    rows = []
    for index in range(outputs.numel()):
        grads = torch.autograd.grad(outputs[index],params,retain_graph=True,allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)]))
    return torch.stack(rows).detach()


def parameter_probe(adapter,loss,step):
    """Return a virtual copy; never modify or serialize a parent Actor."""
    if not 0. < step <= .15:
        raise ValueError("parameter probe step must be in (0,.15]")
    params = tuple(adapter.parameters())
    grads = torch.autograd.grad(loss,params,allow_unused=True)
    vector = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])
    if not torch.isfinite(vector).all():
        raise ValueError("nonfinite Adapter gradient")
    candidate = deepcopy(adapter)
    norm = vector.norm().clamp(min=1e-12)
    with torch.no_grad():
        for target,source,gradient in zip(candidate.parameters(),params,grads):
            if gradient is not None:
                target.copy_(source-float(step)*gradient/norm)
    return candidate,float(vector.norm()),vector.detach()


def assess_probe(baseline,candidate):
    known_risk = (len(baseline.risk_proxy) == len(candidate.risk_proxy)
                  and all(v is not None for v in (*baseline.risk_proxy,*candidate.risk_proxy)))
    gain = (candidate.executed_progress-baseline.executed_progress
            if candidate.executed_progress is not None and baseline.executed_progress is not None else None)
    evidence = {"gain":float(gain) if gain is not None else None,
        "hard_safety_regression":bool(baseline.hard_safe and not candidate.hard_safe),
        "risk_proxy_regression":bool(any(a>b for a,b in zip(candidate.risk_proxy,baseline.risk_proxy))) if known_risk else None,
        "certifiable":bool(baseline.supported and candidate.supported and candidate.hard_safe and known_risk)}
    if not baseline.supported or not candidate.supported or not candidate.hard_safe:
        return {**evidence,"outcome":"unsafe_or_unsupported","positive_safe":False}
    if not known_risk:
        return {**evidence,"outcome":"unknown_risk","positive_safe":False}
    if evidence["risk_proxy_regression"]:
        outcome = "risk_worsened"
    elif np.allclose(candidate.executed_actions,baseline.executed_actions,atol=1e-7,rtol=0.):
        outcome = "absorbed"
    else:
        outcome = "improved" if gain>0. else ("worsened" if gain<0. else "flat_changed_action")
    return {**evidence,"outcome":outcome,"positive_safe":outcome == "improved"}


def audit_action_directions(mean,critic_value,evaluator,goal_direction,epsilon,seed=20261121):
    mean = mean.detach().clone().requires_grad_(True)
    current = critic_value(torch.tanh(mean))
    gradient, = torch.autograd.grad(current,mean)
    if not torch.isfinite(gradient).all():
        raise ValueError("nonfinite executed-value action gradient")
    baseline = evaluator(torch.tanh(mean).detach().cpu().numpy())
    generator = torch.Generator().manual_seed(int(seed))
    random_direction = torch.randn(mean.shape,generator=generator,dtype=mean.dtype).to(mean.device)
    goal = torch.as_tensor(goal_direction,device=mean.device,dtype=mean.dtype)
    probes = []
    for name,direction in (("critic",gradient),("random",random_direction),("goal",goal)):
        action,delta = bounded_probe(mean,direction,epsilon)
        candidate = evaluator(action.detach().cpu().numpy())
        predicted_gain = float((critic_value(action)-current).detach())
        probes.append({"direction":name,"epsilon":float(epsilon),"delta_l2":float(delta.norm()),
            "predicted_gain":predicted_gain,"gradient_norm":float(gradient.norm()),
            **assess_probe(baseline,candidate),"rejection_reasons":candidate.rejection_reasons})
    return probes
