import pandas as pd 
import re
import pandas as pd

KEYS = [
    'model',
    'question_id',
    'question_type',
    'answer_type'
]

def parse_finetuned(s):
    return pd.Series({
        'trainer': 'JS' if 'JS' in s else 'SFT',
        'supervision': 'Blind' if 'Blind' in s else 'GT', 
        'dataset': 'MMStar' if 'MMStar' in s else 'VQA',
        'n': int(re.search(r'n=(\d+)', s).group(1)) if 'n=' in s else None
    })

def extract_size(text):
    if pd.isna(text): return 1.0 # Default fallback
    match = re.search(r'\((\d+(?:\.\d+)?)[Bb]\)', text)
    return float(match.group(1)) if match else 1.0

def result_to_df(res):
    # Extract everything except 'corr_mat'
    data = {k: v for k, v in res.items() if k != 'corr_mat'}
    return pd.DataFrame([data])
    
def calculate_mg(pt, filename=None, answer_similarity=True, categories='answer_type'): 

    pt = pt.pivot_table( 
            index=['model', 'answer_type', 'question_type', 'question_id'], 
            columns=['condition'],   
            values=['correct', 'answer_similarity'],
            aggfunc='mean'
        )  
    pt.columns = [f"{val}_{col}".strip('_').replace(' ', '_') for val, col in pt.columns]
    pt = pt.reset_index()

    vis_col = 'correct' if 'correct' in pt.columns else 'correct_visual'
    sim_vis_col = 'answer_similarity' if 'answer_similarity' in pt.columns else 'answer_similarity_visual'
    pt['Acc_Visual'] = pt[vis_col] 
    pt['Acc_Blind_inst'] = pt['correct_inst_blind']
    pt['MG_Acc'] = pt['Acc_Visual'] - pt['Acc_Blind_inst']

    if answer_similarity: 
        pt['S_Visual'] = pt[sim_vis_col]
        pt['S_Blind_inst'] = pt['answer_similarity_inst_blind']
        pt['MG_S'] = pt['S_Visual'] - pt['S_Blind_inst']
        
        # Delta_Inst logic: Ensure 'correct_blind' exists in your conditions
        if 'correct_blind' in pt.columns:
            pt['Delta_Inst'] = pt['correct_blind'] - pt['correct_inst_blind']
        else:
            pt['Delta_Inst'] = 0 # Fallback if 'blind' condition is missing

    if filename is not None:
        # Standardize export logic: Use the flattened column names
        common_index = ['model']
        
        # 1. Instruction Gains Table
        cols_inst = ['Acc_Visual', 'MG_Acc']
        if 'Delta_Inst' in pt.columns: cols_inst.append('Delta_Inst')
            
        pt.groupby(common_index)[cols_inst].mean().to_latex(
            f'./tables/VQA_{filename}_inst.tex', float_format="%.1f"
        )
        
        # 2. Accuracy by Category
        pt.pivot_table(
            index='model', columns=categories, values=['Acc_Visual', 'MG_Acc'], aggfunc='mean'
        ).to_latex(f'./tables/VQA_{filename}_acc_atype.tex', float_format="%.1f")

        if answer_similarity:
            # 3. Similarity by Category
            pt.pivot_table(
                index='model', columns=categories, values=['S_Visual', 'MG_S'], aggfunc='mean'
            ).to_latex(f'./tables/VQA_{filename}_similarity_atype.tex', float_format="%.1f")
    return pt  

def prep(df_sub, suffix):
    return df_sub[
        KEYS + ['correct', 'answer_similarity']
    ].rename(columns={
        'correct': f'correct_{suffix}',
        'answer_similarity': f'answer_similarity_{suffix}'
    })

def get_merged(df):  
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
    return merged