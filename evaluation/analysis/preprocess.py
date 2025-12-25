import json
import pandas as pd 
from utils.df import *  # result_to_df  
from utils.corr import get_all_cm, get_agreements, transform_corr_table, plot_unified_agreement_heatmap
from utils.quadrants import * # quadrant_proportions_table, median_mg_table   
from utils.score import get_summary  
from utils.plots import scatterplot, get_family, parse_size  
# from utils.df import aggregate_vqa, model_name_map, calculate_mg  


import sys 
sys.path.append('..')
from score_humans import VQAAnswerMapper  # get_vqa_mapper, 


def get_human_results(): 
    with open('/home/work/yuna/HPA/evaluation/scored/humans/human_vqa_per_question.json', 'r') as f:
        data = json.load(f)
        
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        records = [item for sublist in data for item in sublist]
        human_vqa = pd.DataFrame(records)
    else:
        human_vqa = pd.DataFrame(data)
    vqamapper = VQAAnswerMapper()
    vqamapper._load()
    vqa_annot = vqamapper.annotations 

    human_vqa['model'] = "Humans" 
    human_vqa['meta_model'] = "humans" 
    human_vqa['condition'] = "inst blind" 
    human_vqa.rename(columns={"mean_accuracy": 'correct', 'qid': 'question_id'}, inplace=True) 
    human_vqa['correct'] = pd.to_numeric((human_vqa['correct'] * 100).round(1) , errors='coerce')
    human_vqa['answer_similarity'] = pd.to_numeric((human_vqa['answer_similarity'] * 100).round(1) , errors='coerce')
    human_vqa['question_type'] = human_vqa['question_id'].map(lambda qid: vqa_annot[int(qid)]['question_type'] if int(qid) in vqa_annot else None)
    human_vqa['answer_type'] = human_vqa['question_id'].map(lambda qid: vqa_annot[int(qid)]['answer_type'] if int(qid) in vqa_annot else None)
    human_vqa = human_vqa.drop_duplicates(subset=['participant_id', 'question_id'])
    human_vqa = human_vqa[human_vqa['participant_id'] != '13f54aa2_20251204_125847']

    human_vqa.to_csv('/home/work/yuna/HPA/evaluation/analysis/processed_results/human_vqa.csv') 
    return human_vqa 

def get_mg(df): 
    
    pt = df.pivot_table( 
        index=['model', 'question_id', 'answer_type', 'question_type'],  
        columns=['condition'],  
        values=['correct', 'answer_similarity'], 
        aggfunc=['mean'] # , 'count' 
    ) 

    pt['MG'] = (
        pt[('mean', 'correct', '')]
        - pt[('mean', 'correct', 'inst blind')]
    )
    pt['inst_acc'] = pt[('mean', 'correct', 'inst blind')] - pt[('mean', 'correct', 'blind')] 
    # pt.to_csv(f'./tables/mmstar_model_MG_by_qid.csv') 

    pt = pt.dropna(subset=[
        ('mean','correct','inst blind'),
        ('mean','correct','blind')
    ])
    pt.columns = [
        '_'.join([str(i) for i in col if str(i) != '']).strip('_')
        for col in pt.columns.values
    ]
    pt = pt.reset_index()
    return pt  

def human_model_results_n20(model_vqa, human_vqa): # 
    qids= human_vqa.question_id.unique()
    vqa_1k = get_summary('vqa_1k').drop_duplicates(subset=['condition', 'question_id', 'model'])
    vqa_1k['model'] = vqa_1k['model'].map(MODEL_DISPLAY_MAP)  
    model_vqa = vqa_1k[(vqa_1k['question_id'].isin(qids)) & (vqa_1k['condition'] == 'inst blind')].replace(MODEL_DISPLAY_MAP)

    ### combine human and model ! 
    hm = pd.concat([model_vqa, human_vqa]) # just the blind conditions 
    hm.pivot_table(  
        index=['model'], 
        values=['correct', 'answer_similarity'], 
        columns=['answer_type'], 
        aggfunc=['mean'] ,
        margins=True,
        margins_name='Average'
    ).to_latex('./tables/1_VQA_human-pretrained-model_similarity-answer_type.tex', float_format="%.1f")   

    human_avg_by_qid = human_vqa.groupby(['question_type', 'answer_type', 'question_id'])['correct'].mean().reset_index()
    human_avg_by_qid['question_id'] = human_avg_by_qid['question_id'].astype('int')

    return human_avg_by_qid  

def get_quadrants(pt):
    mms=pt.groupby(['question_id', 'question_type', 'answer_type']).mean(numeric_only=True).reset_index() 
    mms = mms[mms['question_id'].isin(qids)]
    mms = pd.merge(human_avg_by_qid, mms, on =['question_type', 'answer_type', 'question_id'], how='inner', suffixes=('_human', '_model')) 
    mms.columns 
    mms["human_correct"] = (mms["mean_correct"] >= 0.5).astype(int)  
    mms["model_correct"] = (mms["mean_correct_inst blind"] >= 0.5).astype(int) 
    mms["new_question_type"] = mms["question_type"].apply(map_question_type) 
    mms["cc_quadrant"] = mms.apply(cc_quadrant, axis=1) 
    mms.groupby("cc_quadrant")["MG"].agg(['count', 'mean']).to_latex('./tables/appendix/vqa-quadrant.tex', float_format="%.1f")  
    mms.groupby(["cc_quadrant", 'answer_type', 'new_question_type'])["MG"].agg(['count', 'mean']).to_latex('./tables/appendix/vqa-quadrant-qtype.tex', float_format="%.1f")  
    
    # pd.merge(hm, mms, on=['question_type', 'question_id'], how='left').to_csv("./tables/examples/vqa.csv")   
    # mms["human_correct"] = (human_avg_by_qid["correct"] >= 50).astype(int)  
    # mms["model_correct"] = (mms["mean_correct_inst blind"] >= 50).astype(int) 
    mms.to_csv('./vqa_quadrants.csv')   
    
    quad_stat = pd.merge(median_mg_table(mms, "MG"), quadrant_proportions_table(mms), on =['cc_quadrant'], suffixes=('_MG_Acc', ''))
    quad_stat.to_latex('./tables/4_VQA_quad_stat.tex', float_format="%.1f")   
    quad_stat  

def get_vqa_corr(human_vqa, pmd): 

    human_cols = human_vqa.participant_id.unique()
    model_cols = pmd.model.unique() 

    print(len(human_cols), len(model_cols), human_vqa.groupby('participant_id')['question_id'].nunique()   )

    res_HH, res_MM, res_HM  = get_agreements(human_vqa, pmd, n_boot=0)  # accuarcy 
    corr_mat = get_all_cm(res_HH, res_MM, res_HM )
    acc_corr = res_HM['corr_mat'].mean().sort_values().reset_index(name='acc_corr') 
    plot_unified_agreement_heatmap(corr_mat, human_cols, model_cols , 'Human–Model Agreement (Accuracy)', '/home/work/yuna/HPA/evaluation/analysis/figures/agreement-acc.png') 
    acc = pd.concat([result_to_df(res_HH), result_to_df(res_MM), result_to_df(res_HM)]) 

    res_HH, res_MM, res_HM  =  get_agreements(human_vqa, pmd, 'answer_similarity', n_boot=0) 
    corr_mat = get_all_cm(res_HH, res_MM, res_HM ) 
    sim_corr = res_HM['corr_mat'].mean().sort_values().reset_index(name='sim_corr') 
    acc = pd.concat([acc, result_to_df(res_HH), result_to_df(res_MM), result_to_df(res_HM)])
    plot_unified_agreement_heatmap(corr_mat, human_cols, model_cols , 'Human–Model Agreement (Emb Similarity)', '/home/work/yuna/HPA/evaluation/analysis/figures/agreement-sim.png') 
    transform_corr_table(acc).to_latex('./tables/3_spearman-interrater-vqa.tex', float_format="%.3f")    

    ### PLOT SCATTER  
    pcorr = pd.merge(acc_corr, sim_corr, on =['model']) 
    df = calculate_mg(vqa_5k, filename='vqa_5k').groupby(['model']).mean(numeric_only=True).reset_index()  
    pcorr = pd.merge(pcorr, df, on=['model']) 
    pcorr = pcorr[['model', 'acc_corr', 'sim_corr', 'MG_Acc', 'MG_S']]
    pcorr['family'] = pcorr['model'].map(get_family)
    pcorr['model_size'] = pcorr['model'].map(parse_size) 

    scatterplot(pcorr, x_col='acc_corr', y_col='MG_Acc', title='accuracy') 
    scatterplot(pcorr, x_col='sim_corr', y_col='MG_S', title='similairty')  