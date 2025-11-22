
from typing import List, Dict, Tuple
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

def load_encoder(model_name: str="all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed. pip install sentence-transformers")
    return SentenceTransformer(model_name)

def embed_texts(encoder, texts: List[str]) -> np.ndarray:
    if len(texts)==0:
        return np.zeros((0,384), dtype=np.float32)
    vecs = encoder.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def cluster_answers(texts: List[str], vecs: np.ndarray, thresh: float=0.8) -> Tuple[List[int], Dict[int, List[int]]]:
    clusters = []
    centers = []
    assign = [-1]*len(texts)
    for i, v in enumerate(vecs):
        if len(centers)==0:
            centers.append(v.copy()); clusters.append([i]); assign[i]=0; continue
        sims = np.array([np.dot(v, c) for c in centers])
        j = int(np.argmax(sims))
        if sims[j] >= thresh:
            clusters[j].append(i)
            centers[j] = (centers[j]*(len(clusters[j])-1) + v) / len(clusters[j])
            assign[i]=j
        else:
            centers.append(v.copy()); clusters.append([i]); assign[i]=len(centers)-1
    return assign, {ci: members for ci, members in enumerate(clusters)}

def canonical_from_clusters(texts: List[str], assign: List[int]) -> Dict[str, str]:
    can_map = {}
    first_by_cluster = {}
    for i, cid in enumerate(assign):
        if cid not in first_by_cluster:
            first_by_cluster[cid] = texts[i]
        can_map[texts[i]] = first_by_cluster[cid]
    return can_map
