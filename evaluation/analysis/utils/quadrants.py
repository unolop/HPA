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

def get_examples(vqq, i, human_vqa, model, target_group='Human-Only'): 
    df = vqq[vqq['cc_quadrant'] == target_group].sort_values(by=['MG'], ascending=False)
    ex = df.iloc[i] 
    qid = ex['question_id']   
    qid = int(qid)

    print(target_group, qid , ex['question'], ex['multiple_choice_answer'] )  

    hdf = human_vqa.loc[
        human_vqa["question_id"] == qid
    ].sort_values("correct", ascending=False)
 
    # print(target_group, ex)
    model_answers = model[model["question_id"] == qid].sort_values(
                                            by=['model', 'finetuned', 'correct', 'processed_ans'], ascending=False)[[ 'model', 'finetuned', 'correct', 'processed_ans']]  
    human_answers = hdf['processed_ans'].values  

    print('Human answers', human_answers, hdf['correct'].values, np.mean(hdf['correct'].values)) 
    print('Model answers') # : human_answers) 

    for i, row in model_answers.iterrows(): 
        print(
            f"{row['model']:<15} | "
            f"{row['finetuned']:<5} | "
            f"Output: {row['processed_ans']} | "
            f"Score: {row['correct']}"
        )
    # model_answers = pd.merge(df, model_answers, on=['question_type', 'answer_type', 'question', 'model', 'question_id', 'condition', 'finetuned', 'multiple_choice_answer', 'image_id'])


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