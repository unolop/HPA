KEYS = [
    'model',
    'question_id',
    'question_type',
    'answer_type'
]

def prep(df_sub, suffix):
    return df_sub[
        KEYS + ['correct', 'answer_similarity']
    ].rename(columns={
        'correct': f'correct_{suffix}',
        'answer_similarity': f'answer_similarity_{suffix}'
    })

def get_merged(df): 
    df = model_vqa.copy()
    df['condition_clean'] = (
        df['condition'].fillna('').astype(str).str.strip().replace({'': 'full'})
    )

    df_full  = prep(df[df['condition_clean'] == 'full'], 'full')
    df_blind = prep(df[df['condition_clean'] == 'blind'], 'blind')
    df_inst  = prep(df[df['condition_clean'] == 'inst blind'], 'inst_blind')

    merged = (
        df_full
        .merge(df_blind, on=KEYS, how='inner')
        .merge(df_inst,  on=KEYS, how='inner')
    )
    merged['inst_acc'] = merged['correct_inst_blind'] - merged['correct_blind']
    merged['MG'] = merged['correct_full'] - merged['correct_inst_blind'] 

    merged['inst_sim'] = merged['answer_similarity_inst_blind'] - merged['answer_similarity_blind']
    merged['MG_sim'] = merged['answer_similarity_full'] - merged['answer_similarity_inst_blind']
    merged