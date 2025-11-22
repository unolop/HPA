
import torch
import torch.nn.functional as F

def js_divergence(p, q, eps=1e-12):
    # p, q: probs over K
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5*(p+q)
    def kl(a,b): return torch.sum(a * torch.log(a/b), dim=-1)
    return 0.5*kl(p,m) + 0.5*kl(q,m)

def ce_to_human(p, target):
    # p: model probs over K; target: human probs over K
    # returns cross-entropy with soft labels
    p = torch.clamp(p, 1e-12, 1.0)
    target = torch.clamp(target, 1e-12, 1.0)
    return - torch.sum(target * torch.log(p), dim=-1)

def brier_loss(p, target):
    return torch.sum((p - target) ** 2, dim=-1)

def human_prior_alignment_loss(model_probs, human_probs, mode="js", w_brier=0.0):
    # model_probs, human_probs: (B,K)
    if mode == "js":
        base = js_divergence(human_probs, model_probs)  # (B,)
    elif mode == "ce":
        base = ce_to_human(model_probs, human_probs)
    else:
        raise ValueError("mode must be 'js' or 'ce'")
    if w_brier>0:
        base = base + w_brier * brier_loss(model_probs, human_probs)
    return base.mean()
