import pandas as pd 
import re
import pandas as pd

KEYS = [
    'model',
    'question_id',
    'question_type',
    'answer_type'
]

model_name_map = {
    'InternVL3_5-1B': 'InternVL 3.5 (1B)',
    'InternVL3_5-2B': 'InternVL 3.5 (2B)',
    'InternVL3_5-4B': 'InternVL 3.5 (4B)',
    'InternVL3_5-8B': 'InternVL 3.5 (8B)',
    'Qwen3-VL-2B-Instruct': 'Qwen3-VL (2B)',
    'Qwen3-VL-4B-Instruct': 'Qwen3-VL (4B)',
    'Qwen3-VL-8B-Instruct': 'Qwen3-VL (8B)',
    'Qwen3-4B':  'Qwen3 (4B)', 
    'Qwen3-8B':  'Qwen3 (8B)',  
    'llava-v1.6-mistral-7b-hf': 'LLaVA-v1.6-Mistral (7B)',
    'llava-1.5-7b-hf': 'LLaVA-v1.5 (7B)',
    'llava-v1.6-vicuna-7b-hf': 'LLaVA-v1.6-Vicuna (7B)',
    'humans': 'Humans'
}

MODEL_DISPLAY_MAP = {
    # ---------- Humans ----------
    "Humans": "Humans",

    # ---------- InternVL ----------
    "InternVL3_5-1B": "InternVL3.5‑1B",
    "InternVL3_5-2B": "InternVL3.5‑2B",
    "InternVL3_5-4B": "InternVL3.5‑4B",
    "InternVL3_5-8B": "InternVL3.5‑8B",

    "InternVL3_5-8B_A1_vqa_gt": "InternVL3.5‑8B | JS‑Blind VQA (GT)",
    "InternVL3_5-8B_A2_vqa_10_blind_inst": "InternVL3.5‑8B | JS‑Blind VQA (n=10)",
    "InternVL3_5-8B_A3_vqa_15_blind_inst": "InternVL3.5‑8B | JS‑Blind VQA (n=15)",
    "InternVL3_5-8B_A4_mmstar_15_blind_inst": "InternVL3.5‑8B | JS‑Blind MMStar (n=15)",
    "InternVL3_5-8B_SFT_vqa_15_blind_inst": "InternVL3.5‑8B | SFT‑Blind VQA",
    "InternVL3_5-8B_SFT_vqa_gt": "InternVL3.5‑8B | SFT‑VQA",

    # ---------- Qwen ----------
    "Qwen3-VL-2B-Instruct": "Qwen3‑VL‑2B",
    "Qwen3-VL-4B-Instruct": "Qwen3‑VL‑4B",
    "Qwen3-VL-8B-Instruct": "Qwen3‑VL‑8B",

    "Qwen3-VL-4B-Instruct_A1_vqa_gt": "Qwen3‑VL‑4B | JS‑Blind VQA (GT)",
    "Qwen3-VL-4B-Instruct_A2_vqa_10_blind_inst": "Qwen3‑VL‑4B | JS‑Blind VQA (n=10)",
    "Qwen3-VL-4B-Instruct_A3_vqa_15_blind_inst": "Qwen3‑VL‑4B | JS‑Blind VQA (n=15)",
    "Qwen3-VL-4B-Instruct_A4_mmstar_15_blind_inst": "Qwen3‑VL‑4B | JS‑Blind MMStar (n=15)",
    "Qwen3-VL-4B-Instruct_SFT_vqa_15_blind_inst": "Qwen3‑VL‑4B | SFT‑Blind VQA",
    "Qwen3-VL-4B-Instruct_SFT_vqa_gt": "Qwen3‑VL‑4B | SFT‑VQA",

    "Qwen3-VL-8B-Instruct_A1_vqa_gt": "Qwen3‑VL‑8B | JS‑Blind VQA (GT)",
    "Qwen3-VL-8B-Instruct_A2_vqa_10_blind_inst": "Qwen3‑VL‑8B | JS‑Blind VQA (n=10)",
    "Qwen3-VL-8B-Instruct_A3_vqa_15_blind_inst": "Qwen3‑VL‑8B | JS‑Blind VQA (n=15)",
    "Qwen3-VL-8B-Instruct_A4_mmstar_15_blind_inst": "Qwen3‑VL‑8B | JS‑Blind MMStar (n=15)",
    "Qwen3-VL-8B-Instruct_SFT_mmstar_15_blind_inst": "Qwen3‑VL‑8B | SFT‑Blind MMStar",
    "Qwen3-VL-8B-Instruct_SFT_vqa_15_blind_inst": "Qwen3‑VL‑8B | SFT‑Blind VQA",

    # ---------- LLaVA ----------
    "llava-v1.6-mistral-7b-hf": "LLaVA‑Mistral‑7B",
    "llava-v1.6-vicuna-7b-hf": "LLaVA‑Vicuna‑7B",
    "llava-1.5-7b-hf": "LLaVA‑1.5‑7B",

    "llava-v1.6-mistral-7b-hf_A1_vqa_gt": "LLaVA‑Mistral‑7B | JS‑Blind VQA (GT)",
    "llava-v1.6-mistral-7b-hf_A2_vqa_10_blind_inst": "LLaVA‑Mistral‑7B | JS‑Blind VQA (n=10)",
    "llava-v1.6-mistral-7b-hf_A3_vqa_15_blind_inst": "LLaVA‑Mistral‑7B | JS‑Blind VQA (n=15)",
    "llava-v1.6-mistral-7b-hf_A4_mmstar_15_blind_inst": "LLaVA‑Mistral‑7B | JS‑Blind MMStar (n=15)",
    "llava-v1.6-mistral-7b-hf_SFT_mmstar_15_blind_inst": "LLaVA‑Mistral‑7B | SFT‑Blind MMStar",
    "llava-v1.6-mistral-7b-hf_SFT_vqa_15_blind_inst": "LLaVA‑Mistral‑7B | SFT‑Blind VQA",
    "llava-v1.6-mistral-7b-hf_SFT_vqa_gt": "LLaVA‑Mistral‑7B | SFT‑VQA",
}


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