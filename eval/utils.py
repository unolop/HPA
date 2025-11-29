import pandas as pd 

def get_answer(question_id):
    adf = pd.DataFrame(annotations)
    df['question_id'] = df['question_id'].astype(int) 
    target_row = df[df['question_id'] == question_id]
    return target_row.iloc[0]['answers']
    
def get_question_id(df, allqs=allqs): # 이게 뭐지 
    allqs['question_id'] = allqs['question_id'].astype('int') 
    allqs = allqs.drop_duplicates(subset=['question'], keep='first') 

    return pd.merge(df, allqs, on=['question'], how='left') 