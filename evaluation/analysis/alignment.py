import torch
import torch.nn.functional as F

def distributional_alignment(
    human_probs: torch.Tensor,
    model_probs: torch.Tensor,
    mode: str = "JS",
    eps: float = 1e-12,
) -> float:
    """
    Compute distributional alignment between human and model distributions.

    Args:
        human_probs: Tensor [K] — human answer distribution (sums to 1)
        model_probs: Tensor [K] — model answer distribution (sums to 1)
        mode: "JS" or "CE"
        eps: numerical stability

    Returns:
        Scalar alignment loss (lower = more aligned)
    """
    h = human_probs.clamp(min=eps)
    m = model_probs.clamp(min=eps)

    if mode == "CE":
        # Cross-entropy H(h, m)
        return float(-(h * torch.log(m)).sum())

    if mode == "JS":
        # Jensen–Shannon divergence
        mid = 0.5 * (h + m)
        js = 0.5 * (h * (torch.log(h) - torch.log(mid))).sum() + \
             0.5 * (m * (torch.log(m) - torch.log(mid))).sum()
        return float(js)

    raise ValueError(f"Unknown mode: {mode}")


def question_alignment(
    human_confidences: list,
    model_counts: list,
    mode: str = "JS",
) -> float:
    """
    Compute alignment for one question.

    Args:
        human_confidences: list[float], e.g. [0.5, 0.3, 0.2]
        model_counts: list[int], number of times model produced each answer
        mode: "JS" or "CE"

    Returns:
        alignment score
    """
    human = torch.tensor(human_confidences, dtype=torch.float32)
    human = human / human.sum()

    model = torch.tensor(model_counts, dtype=torch.float32)
    model = model / model.sum()

    return distributional_alignment(human, model, mode=mode)
