from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2").to('cuda')

def vqa_accuracy(gt_answers, pred):
    """
    gt_answers: list of strings, e.g., ["no", "yes", ...]
    pred: string
    """
    pred = pred.strip().lower()
    
    matches = sum([
        pred == ans.strip().lower()
        for ans in gt_answers
    ])
    
    acc = min(1.0, matches / 3.0)
    return acc


def answer_similarity(gt_answers, pred):
    """gt_answers is a list of strings. pred is a string."""
    pred = pred.strip().lower()
    scores = []
    
    for gta in gt_answers:
        gta = gta.strip().lower()  # string
        emb = encoder.encode([pred, gta])
        similarities = encoder.similarity(emb,emb) 
        scores.append(similarities[1,0])
    return float(np.mean(scores))