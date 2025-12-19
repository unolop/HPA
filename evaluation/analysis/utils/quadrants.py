import pandas as pd

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