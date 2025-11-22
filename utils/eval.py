import numpy as np 

def vqa_accuracy(gt_answers, pred): # pred_answers):
    """
    gt_answers: dict mapping question_id to list of 10 human answers (strings)
    pred_answers: dict mapping question_id to model answer (string)
    Returns: overall VQA accuracy (float)
    """ 
    matches = sum([pred.strip().lower() == ans['answer'].strip().lower() for ans in gt_answers])
    acc = min(1.0, matches / 3.0)
    return acc 

def answer_similarity(encoder, answer, gt:list): 
    avg = [] 
    for gta in gt : 
        gta = gta['answer'] 
        embeddings = encoder.encode([answer, gta]) 
        # print(embeddings.shape) # [3, 384]
        similarities = encoder.similarity(embeddings, embeddings)
        avg.append(similarities[1,0]) 
        # print(similarities) 

    return np.mean(avg) 