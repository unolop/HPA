import re 

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


def get_examples(quad, df, mms):  
    
    both_wrong = mms[mms["cc_quadrant"] == quad]
    target_ids = both_wrong.sort_values("MG_Acc").head(2)["question_id"].tolist()
    result_df = df[df['question_id'].isin(target_ids)].copy()
    print(target_ids, len(result_df))
    result_df['quadrant'] = quad
    # result_df = result_df.dropna(axis=1)

    return result_df
