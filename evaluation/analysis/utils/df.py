
import re
import pandas as pd
import ast
import numpy as np  

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

def parse_ci(x):
    # Case 1: NumPy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # Case 2: Python list or tuple
    if isinstance(x, (list, tuple)):
        return list(x)

    # Case 3: String
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, np.ndarray):
                return parsed.tolist()
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            return np.nan

    return np.nan

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
