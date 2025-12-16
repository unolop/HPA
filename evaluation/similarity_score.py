import numpy as np 

def compute_similarity(gt: str, pred: str, encoder) -> float:
    if encoder is None or not gt or not pred:
        return 0.0

    emb = encoder.encode(
        [pred.strip(), gt.strip()],
        # normalize_embeddings=True
    )

    sim = float((emb[0] @ emb[1]))
    return float(np.clip(sim, -1.0, 1.0)) 