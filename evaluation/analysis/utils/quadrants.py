import re 
import numpy as np 
import pandas as pd 

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
    model_answers = model[model["question_id"] == qid].sort_values(by=['correct'], ascending=False)  
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
    model_answers = pd.merge(df, model_answers, on=['question_type', 'answer_type', 'question', 'model', 'question_id', 'condition', 'finetuned', 'multiple_choice_answer', 'image_id'])
    return {
        "target_group": target_group , 
        "question_id": qid, 
        "question": ex['question'], 
        "model": model_answers , 
        "human": hdf, 
        'Human answers': human_answers, 
        "human_acc": hdf['correct'].values, 
    }  

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

def cc_quadrant(row):
    if row["human_correct"] == 1 and row["model_correct"] == 1:
        return "Shared Correct"
    elif row["human_correct"] == 1 and row["model_correct"] == 0:
        return "Human-Only"
    elif row["human_correct"] == 0 and row["model_correct"] == 1:
        return "Model-Only"
    else:
        return "Shared Wrong"

def quadrants_qusetion_types(mms): 
    # by questions types  
    qt_cc = (
        mms.groupby(["question_type", "cc_quadrant"])
        .size()
        .reset_index(name="count")
    )

    qt_cc["fraction"] = (
        qt_cc["count"] /
        qt_cc.groupby("question_type")["count"].transform("sum")
    )
    qt_cc_table = qt_cc.pivot(
        index="question_type",
        columns="cc_quadrant",
        values="fraction"
    ).fillna(0) 
    
    return qt_cc_table 
    
def quadrant_proportions_table(df):
    """
    Returns fraction of questions in each blind correctness quadrant.
    """
    counts = (
        df["cc_quadrant"]
        .value_counts()
        .rename("count")
        .reset_index()
        .rename(columns={"index": "Blind Correctness Pattern"})
    )

    total = counts["count"].sum()
    counts["fraction"] = counts["count"] / total

    return counts # .sort_values("Blind Correctness Pattern")


def median_mg_table(df, mg_col="model_MG_acc_q"):
    """
    Computes mean and median grounded performance per blind correctness quadrant.
    """
    summary = (
        df.groupby("cc_quadrant")[mg_col]
          .agg(
              mean="mean",
              median="median",
              n="count"
          )
          .reset_index()
    )

    return summary