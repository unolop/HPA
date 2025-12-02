from sentence_transformers import SentenceTransformer
from processor import extract_mc_choice, get_conditions 

encoder = SentenceTransformer("all-MiniLM-L6-v2").to('cuda')


def read_file(filepath, evaluate=True): 
    # with open(filepath, 'r', encoding='utf-8') as f:
    
    df = pd.read_json(filepath, lines=True)  
    model_full, dataset, condition = get_conditions(filepath)
    df['model_full'], df['dataset'], df['condition'] = model_full, dataset, condition

    if dataset == 'spubench' or dataset == "mmstar": 
        try : 
            df["mc_choice"] = df["output"].apply(lambda x: extract_answer_from_output(x))
            df["correct"] = df["correct"].apply(lambda x: True if x is True else False)
            df["correct"] = df["correct"].astype(int)
        except Exception as e: 
            print(f'{e} cannot process {filepath}')

    else: 
        PostProcessor

    return df 

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