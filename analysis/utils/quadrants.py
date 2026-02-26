import re 
import numpy as np 
import pandas as pd 
from scipy.stats import mannwhitneyu


### SANITY CHECKS 
def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    return (np.sum(x[:,None] > y) - np.sum(x[:,None] < y)) / (nx * ny)
    
def bootstrap_ci(mms, n=10000):

    ho = mms.loc[mms.cc_quadrant == 'Human-Only', 'MG']
    sc = mms.loc[mms.cc_quadrant == 'Shared Correct', 'MG']

    u, p = mannwhitneyu(ho, sc, alternative='greater')
    delta = cliffs_delta(ho.values, sc.values)
    bootstrap_ci(ho.values, sc.values) 
    
    diffs = []
    for _ in range(n):
        xs = np.random.choice(ho.values, size=len(ho.values), replace=True)
        ys = np.random.choice(sc.values, size=len(sc.values), replace=True)
        diffs.append(xs.mean() - ys.mean())
    return np.percentile(diffs, [2.5, 97.5])


def cc_quadrant(row):
    if row["human_correct"] == 1 and row["model_correct"] == 1:
        return "Shared Correct"
    elif row["human_correct"] == 1 and row["model_correct"] == 0:
        return "Human-Only"
    elif row["human_correct"] == 0 and row["model_correct"] == 1:
        return "Model-Only"
    else:
        return "Shared Wrong"

def get_examples(vqq, finetuned_models, human_vqa, qid=None): 

    vqq = vqq[vqq['model'].isin(['InternVL 3.5 (8B)', 'Qwen3‑VL (8B)', 'LLaVA‑Mistral (7B)'])]

    if qid is None: 
        qid = np.random.choice(vqq.question_id.unique())
    print('QID', int(qid), vqq.category.unique()[0])
    df = vqq[vqq['question_id'] == qid]
    hdf = human_vqa.loc[
        human_vqa["question_id"] == qid
    ].sort_values("correct", ascending=False) 
    human_answers = hdf['answer_normalized'].values

    print('Question', df['question'].unique()[0], 'Answer:', df['answer'].unique()[0])
    print('Humans', human_answers, np.mean(hdf['correct'].values))  # hdf['correct'].values,  
    print('Model answers') # : human_answers) 

    # finetuned_models = mm.model_blind  
    finetuned_models = finetuned_models[(finetuned_models['finetuned'] != 'Pretrained') & (finetuned_models['question_id'] == qid)]

    for i, row in df.sort_values(by=['model', 'finetuned'], ascending=False).iterrows():  
        print(  
            # f"{target_group} {qid} {ex['question']} {ex['answer']}\n"
            f"{row['model']:<15} | "
            f"{row['cc_quadrant']:<15} | " 
            f"{row['finetuned']:<5} | "
            f"Output: {row['extracted_choice']} | "
            f"Score: {row['correct_model']}"
        )  

def map_question_type(question: str) -> str:
    try: 
        q = question.lower().strip()
        # Counting
        if re.match(r"^how (many|much)\b", q):
            return "Number"

        # Yes / No
        if re.match(r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b", q):
            return "Yes/No"

        # Person
        if q.startswith("who"):
            return "Person"

        # Reason
        if q.startswith("why"):
            return "Reason"

        # Attribute / Object
        if q.startswith(("what", "which")):
            return "Attribute/Object"

    except Exception as e : 
        print(question, 'cannot be mapped')
    return "Other"