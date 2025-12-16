import numpy as np 
from scipy.stats import pearsonr 
from math import atanh, tanh, sqrt 
from sklearn.metrics.pairwise import cosine_similarity

def get_pearsonr_correlation(values_dict):   
    """
    dictionary of 2 keys each with an array of same legnth
    """  
    x_name,y_name = values_dict.keys()  
    x = values_dict[x_name]
    y = values_dict[y_name] 

    n = len(x) 
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))  
    r_val, p_val = pearsonr(x, y) 

    # 95% CI via Fisher z
    z = atanh(r_val)
    se = 1 / sqrt(n - 3)
    z_crit = 1.96  # 95% CI

    ci_low = tanh(z - z_crit * se)
    ci_high = tanh(z + z_crit * se)
    return {
        f"mean_{x_name}": x_mean, 
        f"mean_{y_name}": y_mean,  
        "method": "pearson",
        "r": round(float(r_val), 3),
        "p_value": round(float(p_val), 4),
        "n": n,
        "ci_95": [round(float(ci_low), 3), round(float(ci_high), 3)]
    } 


def get_encoder():
    """Lazy load sentence transformer."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2").to('cuda')
    except:
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")  # CPU fallback
        except:
            return None


def compute_similarity(gt: str, pred: str, encoder) -> float:
    if encoder is None or not gt or not pred:
        return 0.0

    emb = encoder.encode(
        [pred.strip(), gt.strip()],
        # normalize_embeddings=True
    )

    sim = float((emb[0] @ emb[1]))
    return float(np.clip(sim, -1.0, 1.0)) 