import numpy as np 
import json
import pandas as pd 
from utils.df import *  # result_to_df   aggregate_vqa, model_name_map, calculate_mg   
from utils.corr import get_all_cm, get_agreements, transform_corr_table, plot_unified_agreement_heatmap
from utils.quadrants import * # quadrant_proportions_table, median_mg_table   
from utils.score import get_summary, extract_mc_choice, MODEL_DISPLAY_MAP
from utils.vqa import score_number, score_yes_no, VQAAnswerMapper, vqa_accuracy , PostProcessor 
from utils.ac1 import interrater_agreement_ac1 
from utils.plots import scatterplot, get_family, parse_size  

import sys 
sys.path.append('..')

def test_human_data(human_vqa): 
        
    # with open("/home/work/yuna/HPA/data/training/s1_text/cleaned_n15_text.json", 'r') as f:
    with open("/home/work/yuna/HPA/data/training/s1_choice/cleaned_n15_choice.json", 'r') as f:
        data = json.load(f)
    pids = pd.DataFrame(data).participant_id.unique() 
    pids = np.append(pids, "2e184452_20251205_141511_cleaned")

    trainh = human_vqa[human_vqa['participant_id'].isin(pids)] 
    human_vqa = human_vqa[~human_vqa['participant_id'].isin(pids)] 
    print(len(trainh.participant_id.unique()) , len(human_vqa.participant_id.unique()))  
    
    return human_vqa 

def get_training_regime(pcorr): 
    pcorr['model'] = pcorr['model'].map(MODEL_DISPLAY_MAP)   # TODO needs to be fixed no manaual mapping  

    pcorr = pcorr[
        pcorr['model'].notna() &
        pcorr['model'].apply(lambda x: isinstance(x, str))
    ].copy()

    pcorr['model'] = pcorr['model'].astype(str)
    pcorr['family'] = pcorr['model'].map(get_family)
    pcorr['model_size'] = pcorr['model'].map(parse_size).astype(int)
    pcorr["finetuned"] = (
        pcorr['model']
        .str.split('|').str[1]
        .str.replace("\u2011", "-", regex=False)
        .str.replace("\u2013", "-", regex=False)
        .str.replace("\u2014", "-", regex=False)
        .str.strip()
    ).fillna('Pretrained')

    pcorr['model'] = (
        pcorr['model']
        .str.split('|')
        .str[0]
        .str.strip()
    )
    return pcorr

class SpubenchResults(): 
    def __init__(self):
        self.annot = pd.read_json("/home/work/yuna/HPA/dataset/annotation.json").reset_index()
        self.annot.rename(columns={'index': "pid"}, inplace=True)
        df = self.get_df()  
        self.results = df[df['condition'] == ''] 
        # self.results.groupby(['model'])['correct'].mean().reset_index().to_csv('./spubench.csv', encoding="utf-8-sig")

    def get_df(self): 
        df = get_summary(dataset='spubench')
        df = pd.merge(df, self.annot, on="pid", how="left", suffixes=("", "_annot"))
        # df = df.loc[:, ~df.columns.duplicated()]
        df = df[df['condition'] == '']
        df['dataset'] = 'Spubench'  
        # df = df.dropna(axis=0, subset=['model'])
        df["mc_choice"] = df["output"].apply(extract_mc_choice)
        df["answer"] = df["answer"].fillna(df["answer_annot"])
        df["spurious_correlation_type"] = df["spurious_correlation_type"].fillna(df["spurious_correlation_type_annot"])
        df["correct"] = (
            (df["mc_choice"].str.strip().str[0].str.upper()
            == df["answer"].str.upper())
        ) * 100 
        df = self.get_bias(df) 
        df = df.drop_duplicates(['model', 'pid']) 
        df = get_training_regime(df) 
        return df # .dropna(axis=1, how="all") 

    def get_bias(self,df):
        # df['spurious_correlation_type'] = df['spurious_correlation_type'].apply(
        #     lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        # )
        df = df.explode('spurious_correlation_type') 
        df['bias'] = df['spurious_correlation_type'].map({
                    'Shape':'Sha.', 'Background':'BG', 'Co-occurring Objects': "CO", 
                    'Orientation': "Ori.",
                    'Colorization':'Col.', 
                    'Texture and Noise': 'TN', 
                    'Lighting and Shadows': 'LS',
                    'Relative Size': 'RS', 
                    'Perspective and Angle': 'PA', 
                    'Context': 'CO', 
                    'Packaging': 'PA',
                    'Position': 'Pos.', 
                    'Material': 'Mat.', 
                    'Size': 'Size', 
                    'Appearance': 'App.', 
                    'Motion Blur': 'MB'  
                })

        return df 
    
    def finetuning_effect(self, df ): 

        # merged with delta 
        bs = df[df["finetuned"] == 'Pretrained']
        ft = df[df["finetuned"] != 'Pretrained'] 

        df = pd.merge(bs,ft, on=['model', 'pid', "bias"], how='right', suffixes=("_baseline", "")).dropna(axis=1) 
        df['delta'] = df['correct'].astype(float) - df['correct_baseline'].astype(float)
        # df = df[['model', 'finetuned', 'pid', 'bias', 'delta']]

        pt = df.pivot_table(
            index=['model', 'finetuned',], 
            columns=['bias'] , 
            values=['delta']).dropna(axis=1).reset_index(level=0, drop=True) 
        # pt.columns = pt.columns.droplevel(0)
        # pt.to_latex('./tables/spubench-finetuned-delta.tex', float_format="%.3f")   
             
        return df, pt   

class MMStarResults(): 
    def __init__(self, results_path='/home/work/yuna/HPA/evaluation/scored/humans/human_mc_per_question.json'): 

        with open(results_path) as f:
            raw_data = json.load(f) 

        human_mc = pd.DataFrame(raw_data).rename(columns={'\ufeffquestion_num': 'question_num'}) 
        human_mc['model'] = 'Humans'
        human_mc['condition'] = 'inst blind' 
        self.human = human_mc[human_mc['participant_id']!= '13f54aa2_20251204_125847'] 
        self.human['correct'] = self.human['correct'] * 100  
        self.human = self.human.rename(columns={'pid': 'question_id'}) #   = self.clean_df(human_mc)
        self.qids = self.human.question_id.unique()    
        # self.model = self.get_mg(blind_model[blind_model['question_id'].isin(self.qids)]) 
        self.test_human = test_human_data(self.human) 

        ### MODEL RESULTS 
        model_mc = get_summary('mmstar').drop_duplicates(subset=['model', 'pid', 'condition'])
        model_mc = model_mc.rename(columns={'pid': 'question_id'})  
        model_mc= get_training_regime(model_mc)

        test_model = model_mc[~model_mc['question_id'].isin(self.qids)]
        test_model = self.clean_df(test_model) 
        self.test_model = test_model[test_model['condition'] == '']

        model_mc = model_mc[model_mc['question_id'].isin(self.qids) ]     
        self.model = model_mc[model_mc['condition'] == 'inst blind'] 
        self.model_mg = self.get_mg(model_mc, filename='model') 
        self.test_model_mg = self.get_mg(test_model, filename='test_model_only') 
        # model_mc = self.clean_df(model_mc)

    def clean_df(self, df): 
        mmstar_cat_map = {'coarse perception': "CP", 'fine-grained perception': "FP",
                'instance reasoning': "IR", 'logical reasoning': "LR", 'math': "Math",
                'science & technology': 'S&T'} 
        df['category'] = df['category'].map(mmstar_cat_map)
        df['question_id'] = df['question_id'].astype('Int64') 

        return df 

    def get_quadrants(self): 
        mms= pd.merge(self.model, self.human.groupby(['question_id', 'l2_category', 'category']).mean(numeric_only=True).reset_index(), 
                        on=['question_id', 'l2_category', 'category'], how='left', suffixes=("_model", "_human")) 
        mms["human_correct"] = (mms["correct_human"] >= 50).astype(int)  
        mms["model_correct"] = (mms["correct_model"] >= 50).astype(int) 
        mms["cc_quadrant"] = mms.apply(cc_quadrant, axis=1) 
        mms = pd.merge(self.model_mg, mms, on=['question_id', 'category', 'finetuned', 'model'], how='left')  # .groupby("cc_quadrant")["MG"].agg(['count', 'mean']).to_latex('./tables/appendix/vqa-quadrant.tex', float_format="%.1f")  
        mms['dataset'] = "MMStar" 
        # mms.groupby("cc_quadrant")["MG"].agg(['count', 'mean']).to_latex('./tables/appendix/vqa-quadrant.tex', float_format="%.1f")
        
        return mms

    def get_mg(self, df, filename='inst-blind_only'): 
        pt = df.pivot_table(
            index=['model', 'finetuned', 'category', 'l2_category', 'question_id'],
            values='correct',
            columns='condition',
            aggfunc='mean'
        ).reset_index()

        pt['MG'] = pt[''] - pt['inst blind']
        pt['Delta_Inst'] = pt['blind'] - pt['inst blind']

        pt[pt['question_id'].isin(self.qids)].pivot_table(
            index=['model'],
            values='MG',
            columns=['category'], # , 'l2_category'
            aggfunc='mean'
        ).to_latex(f'/home/work/yuna/HPA/evaluation/analysis/tables/MMSTAR_MG_{filename}-by-category.tex', float_format="%.1f")
        # pt.groupby(['model']).mean(numeric_only=True).to_latex(f'./tables/appendix/mmstar-mg_{filename}.tex', float_format="%.1f")

        return pt 

    def finetuning_effect( 
        self,
        pt, VALUE_COL = 'correct' 
    ):
        KEY_COLS = ['model', 'category', 'l2_category', 'question_id']
        baseline = pt[pt['finetuned'] == "Pretrained"].copy()
        finetuned = pt[pt['finetuned'] != "Pretrained"].copy()
        
        df = finetuned.merge(
            baseline[KEY_COLS + [VALUE_COL]],
            on=KEY_COLS,
            how='inner',
            suffixes=('', '_baseline')
        )
        df['dataset'] = 'MMStar'  
        df['delta'] = (
            df[VALUE_COL].astype(float) -
            df[f'{VALUE_COL}_baseline'].astype(float)
        )
        return df # .drop_duplicates(subset=KEY_COLS) 

    def combine_mh(self, metrics='correct'): 
        
        grouping = ['category', 'l2_category', 'question_id', 'question', 'answer'] 
        # mm = pd.merge(self.model, self.human, on=grouping, how='left', suffixes=('_model', '_human')).groupby().agg({'correct_human': 'mean', 'correct_model': 'mean'}).reset_index()
        # mms = pd.merge(mms, pt.groupby(['category', 'l2_category', 'question_id']).agg({'MG': 'mean'}).reset_index(), on=['category', 'l2_category', 'question_id']) # 'model', , 'correct_model': 'mean'
        # print('combined model and human questions example: ', mm.iloc[0]['question_model'], mm.iloc[0]['question_human']) # .model.unique()   

        human_acc = self.human.pivot(
                index=grouping, 
                columns="participant_id",
                values=metrics
            )
        model_acc = self.model.pivot_table( 
            index=grouping,  
            columns="model",
            values=metrics, 
            aggfunc='mean'
        ) 
        answers_pivot = human_acc.join(model_acc, how="inner").reset_index() 

        return answers_pivot    

    def get_corr(self, answers_pivot, n_boot=200): 
        
        ### GET CATEGORY WISE CORRELATION 
        category_corr = [] 
        for category in answers_pivot.category.unique():  
            res = interrater_agreement_ac1(
                answers_pivot[answers_pivot['category'] == category],
                grp1=self.human_list,
                grp2=self.model_list,
                n_boot=0,
                ) 
            res['category'] = category 
            category_corr.append(res)
        category_corr = pd.DataFrame(category_corr).dropna(axis=0, subset=['mean_r']).dropna(axis=1).drop(columns=['corr_mat', 'title', 'metric'])
        category_corr.to_latex(f"/home/work/yuna/HPA/evaluation/analysis/tables/appendix/mmstar-ac1-category.tex", float_format="%.1f")  

        model_corr = [] 
        for i, cc in category_corr.iterrows(): 
            corr_mat=cc['corr_mat'].T.reset_index().melt(
                id_vars=['model'], var_name='subj', value_name='corr')    
            corr_mat['category'] = cc['category']
            model_corr.append(corr_mat) 
        model_corr= pd.concat(model_corr).dropna()
        model_corr.groupby(['model', 'category'])['corr'].mean().reset_index().to_csv('./corr-mmstar-MH-cat.csv', encoding="utf-8-sig")
 
        res_HH = interrater_agreement_ac1(answers_pivot, self.human_list, self.human_list, title="Human–Human", plot=False ) 
        res_MM = interrater_agreement_ac1(answers_pivot, self.human_list, self.model_list, title="Human–Model", plot=False ) 
        res_HM = interrater_agreement_ac1(answers_pivot, self.model_list, self.model_list, title="Model–Model",  plot=False )  
        
        corr_mat = get_all_cm(res_HH, res_MM, res_HM )
        # res['corr_mat'] = res['corr_mat'].fillna(1)
        plot_unified_agreement_heatmap(
            corr_mat, self.human_list, self.model_list , 
            f'Human–Model Agreement (AC1)', 
            f'/home/work/yuna/HPA/evaluation/analysis/figures/mmstar-AC1_{len(self.model_list)}.png') 
        
        if n_boot:  
            transform_corr_table(pd.concat([result_to_df(res_HH), result_to_df(res_MM), result_to_df(res_HM)])).to_latex(
                '/home/work/yuna/HPA/evaluation/analysis/tables/mmstar_ac1.tex', float_format="%.3f")    
        
        acc_corr = res_HM['corr_mat'].mean().sort_values().reset_index(name='corr') 
        return acc_corr 


class VQAResults():  
    def __init__(self, results_path='/home/work/yuna/HPA/evaluation/scored/humans/human_vqa_per_question.json'): 
        
        with open(results_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            records = [item for sublist in data for item in sublist]
            human_vqa = pd.DataFrame(records)
        else:
            human_vqa = pd.DataFrame(data) 

        vqa_annot = VQAAnswerMapper() 
        vqa_annot._load()
        self.vqa_annot = vqa_annot.annotations 
        self.human = self.clean_df(human_vqa.rename(columns={'qid': 'question_id'}) )
        self.human = self.new_score(self.human)
        # self.human = self.score_answer_types(self.human, answer_column="answer_normalized", gt_column="visual_gt") 
        self.qids= self.human.question_id.unique().astype(int)
        self.test_human = test_human_data(self.human) 
        
        # get model intersection subset 
        vqa_1k = get_summary('vqa_1k').drop_duplicates(subset=['condition', 'question_id', 'model'])
        vqa_1k = self.new_score(vqa_1k, 'output', 'multiple_choice_answer')
        vqa_1k = get_training_regime(vqa_1k) 
        
        self.model = vqa_1k[(vqa_1k['question_id'].isin(self.qids)) & (vqa_1k['condition'] == 'inst blind')] # .replace(MODEL_DISPLAY_MAP)
        vqa_5k = get_summary('vqa_5k').drop_duplicates(subset=['condition', 'question_id', 'model', 'strategy', 'blind', 'trained_dataset'])
        vqa_5k = self.new_score(vqa_5k, "output", "multiple_choice_answer") 
        self.test_model = get_training_regime(vqa_5k)  
        
        self.model_mg = self.get_mg(vqa_1k[(vqa_1k['question_id'].isin(self.qids))], 'subset')  
        self.test_model_mg = self.get_mg(self.test_model, '5k-test')   
        self.quadrants = self.get_quadrants()
    
    def finetuning_effect(self, test_model):  
        df = test_model[test_model['condition'] == ''] 
        bs = df[df['finetuned'] == 'Pretrained']
        ft = df[df['finetuned'] != 'Pretrained']
        df = pd.merge(bs, ft, on =['model', 'answer_type', 'model_size', 'question_id'], suffixes=['_baseline', ''], how ='inner') 
        df['dataset'] = 'VQA'
        df['answer_similarity'] = df['answer_similarity'] - df['answer_similarity_baseline']
        df['accuracy'] = df['correct'] - df['correct_baseline']
        
        df = df.melt(
                    id_vars=[col for col in df.columns if col not in ['answer_similarity', 'accuracy']],
                    value_vars=['answer_similarity', 'accuracy'],
                    var_name='score_type',
                    value_name='delta' 
                    )
        return df
    
    def get_quadrants(self): 
        # model_vqa.groupby(['question_id', 'answer_type', 'question_type']).mean(numeric_only=True).reset_index()
        mms= pd.merge(self.model, self.human.groupby(['question_id', 'answer_type', 'question_type']).mean(numeric_only=True).reset_index(), 
                        on=['question_type', 'question_id', 'answer_type'], how='left', suffixes=("_model", "_human")) 
        mms = pd.merge(mms, self.model_mg.groupby(['model', 'finetuned', 'question_id'])['MG'].mean(), on=['model', 'finetuned', 'question_id'], how='left') # .head() # , 'question_id', 'answer_type', 'question_type'
        
        # mms = pd.merge(model, human_vqa.groupby(['question_id', 'answer_type', 'question_type']).mean(numeric_only=True).reset_index(), 
        #             on=['question_type', 'question_id', 'answer_type'], how='left', suffixes=("_model", "_human")) 
        mms["human_correct"] = (mms["correct_human"] >= 50).astype(int)  
        mms["model_correct"] = (mms["correct_model"] >= 50).astype(int) 
        mms["cc_quadrant"] = mms.apply(cc_quadrant, axis=1) 

        # mms.groupby("cc_quadrant")["MG"].agg(['count', 'mean']).to_latex('./tables/appendix/vqa-quadrant.tex', float_format="%.1f")
        return mms 

    def get_examples_quadrants(self): 
        from utils.quadrants import get_examples 

        for i in [0, 1, -2, -1]: 
            for quad in self.quadrants.cc_quadrant.unique(): 
                ex = get_examples(self.quadrants, i, self.human, self.model, quad)  

                ex["model"].to_csv(
                    f"/home/work/yuna/HPA/evaluation/analysis/quadrant_examples/"
                    f"{ex['target_group']}_{i}_model.csv"
                )

                ex["human"].to_csv(
                    f"/home/work/yuna/HPA/evaluation/analysis/quadrant_examples/"
                    f"{ex['target_group']}_{i}_human.csv"
                )

    def get_pivots(self): 
        hm = pd.concat([self.model, self.human]) 

        hm.pivot_table(  
            index=['model'], 
            values=['correct', 'answer_similarity'], 
            aggfunc=['mean'] ,
        ).to_latex('./tables/appendix/1_VQA_human-finetuned-model_similarity.tex', float_format="%.1f")    

        hm.pivot_table(  
            index=['model'], 
            values=['correct', 'answer_similarity'], 
            columns=['answer_type'], 
            aggfunc=['mean'] ,
        ).to_latex('./tables/appendix/1_VQA_human-finetuned-model_similarity-answer_type.tex', float_format="%.1f")    

    def new_score(self, human_vqa, ans_col='answer_normalized', gt_col='gt_answers'):
        print('before' , np.mean(human_vqa['correct']))
        pp = PostProcessor() 
        human_vqa['processed_ans'] = human_vqa[ans_col].apply(pp.postprocess_answer)
        human_vqa['correct'] = human_vqa.apply(
            lambda row: max(
                row['correct'] if pd.notna(row.get('correct')) else 0,
                vqa_accuracy(row["processed_ans"], row[gt_col]) * 100 
            ),
            axis=1
        ) 
        print('after ', np.mean(human_vqa['correct']))

        return human_vqa 

    def clean_df(self, human_vqa): 
        
        human_vqa['model'] = "Humans" 
        human_vqa['meta_model'] = "humans" 
        human_vqa['condition'] = "inst blind" 
        human_vqa.rename(columns={"mean_accuracy": 'correct', 'qid': 'question_id'}, inplace=True) 
        # map categories 
        human_vqa['question_type'] = human_vqa['question_id'].map(lambda qid: self.vqa_annot[int(qid)]['question_type'] if int(qid) in self.vqa_annot else None)
        human_vqa['answer_type'] = human_vqa['question_id'].map(lambda qid: self.vqa_annot[int(qid)]['answer_type'] if int(qid) in self.vqa_annot else None)
        human_vqa['question_id'] = human_vqa['question_id'].astype('int') 
        human_vqa["new_question_type"] = human_vqa["question_type"].apply(map_question_type) 

        human_vqa = human_vqa.drop_duplicates(subset=['participant_id', 'question_id'])
        human_vqa = human_vqa[human_vqa['participant_id'] != '13f54aa2_20251204_125847']
        human_vqa['correct'] = human_vqa['correct'] * 100 
        human_vqa.to_csv('/home/work/yuna/HPA/evaluation/analysis/human_vqa.csv') 
        return human_vqa 

    def get_mg(self, df, filename='inst-blind_only'): 
        df['new_question_type'] = df["question_type"].apply(map_question_type) 

        pt = df.pivot_table(
            index=['model', 'finetuned', 'answer_type', 'question_type', 'new_question_type', 'question_id'],
            values='correct',
            columns='condition',
            aggfunc='mean'
        ).reset_index()

        pt['MG'] = pt[''] - pt['inst blind']
        pt['Delta_Inst'] = pt['blind'] - pt['inst blind']

        pt.pivot_table(
            index=['model'],
            values=['MG'],
            columns=['answer_type'],
            aggfunc='mean'
        ).to_latex(f'/home/work/yuna/HPA/evaluation/analysis/tables/VQA_MG_{filename}-by-answer_type.tex', float_format="%.1f")
        # pt.groupby(['model']).mean(numeric_only=True).to_latex(f'./tables/appendix/mmstar-mg_{filename}.tex', float_format="%.1f")

        return pt 

    ### UTILS 
    def make_numeric_cols(pm): 
        cols = ['correct_model', 'answer_similarity_model', 'correct_human', 'answer_similarity_human']
        for col in cols:
            pm[col] = pd.to_numeric(pm[col], errors='coerce')

        pm = pm.groupby(['question_id'])[cols].mean() 
        len((pm['correct_human'])), len(pm['correct_model']), len(pm['answer_similarity_human']), len(pm['answer_similarity_model'] ) 
        return pm 

    def melt_score(hm): 
        hm_melted = hm.melt(
                    id_vars=[col for col in hm.columns if col not in ['correct', 'answer_similarity']],
                    value_vars=['correct', 'answer_similarity'],
                    var_name='score_type',
                    value_name='value'
                    )
        print(hm_melted.columns)
        return hm_melted   

class AlignmentAnalysis:
    def __init__(self, dataset):
        if dataset == 'vqa': 
            results = VQAResults() 
            self.human = results.human 
            self.model = results.model 
            self.test_model = results.test_model 
            self.df_qid = self.combine_vqa() 
            df = calculate_mg(self.test_model, filename='vqa_5k').groupby(['model']).mean(numeric_only=True).reset_index()  

            ### PLOT SCATTER  
            pcorr = self.get_corr('accuracy')  # pcorr = pd.merge(acc_corr, sim_corr, on =['model']) 
            pcorr = pd.merge(pcorr, df, on=['model']) 
            pcorr = pcorr[['model', 'acc_corr', 'sim_corr', 'MG_Acc', 'MG_S']]
            pcorr['family'] = pcorr['model'].map(get_family)
            pcorr['model_size'] = pcorr['model'].map(parse_size)  
            scatterplot(pcorr, x_col='acc_corr', y_col='MG_Acc', title='accuracy') 
            
            sim_corr = self.get_corr()   
            sim_corr = pd.merge(pcorr, df, on=['model']) 
            sim_corr = pcorr[['model', 'acc_corr', 'sim_corr', 'MG_Acc', 'MG_S']]
            sim_corr['family'] = pcorr['model'].map(get_family)
            sim_corr['model_size'] = pcorr['model'].map(parse_size)  
            scatterplot(sim_corr, x_col='sim_corr', y_col='MG_S', title='simliarity')  

            pcorr = pd.merge(acc_corr, sim_corr, on =['model']) 
            self.pcorr   

        else: 
            results = MMStarResults() 
            self.human = results.human 
            self.model = results.model 
            self.test_model = results.test_model 
            self.df_qid = self.combine_vqa() 

    def get_corr(self, model_cols=None, metric='answer_similarity', n_boot=0): 

        human_cols = self.human.participant_id.unique()
        if model_cols is None : 
            model_cols = self.model['model'].unique() 
        # print(len(human_cols), len(model_cols), human_vqa.groupby('participant_id')['question_id'].nunique()   )

        res_HH, res_MM, res_HM  = get_agreements(self.human, self.model, n_boot=0)  # accuarcy 
        corr_mat = get_all_cm(res_HH, res_MM, res_HM )
        acc_corr = res_HM['corr_mat'].mean().sort_values().reset_index(name=f'{metric}_corr') 

        if n_boot:  
            transform_corr_table(pd.concat([result_to_df(res_HH), result_to_df(res_MM), result_to_df(res_HM)])).to_latex(
                '/home/work/yuna/HPA/evaluation/analysis/tables/3_spearman-interrater-vqa.tex', float_format="%.3f")    
        
        plot_unified_agreement_heatmap(
            corr_mat, human_cols, model_cols , 
            f'Human–Model Agreement ({metric})', 
            f'/home/work/yuna/HPA/evaluation/analysis/figures/agreement-{metric}_{len(model_cols)}.png') 
        
        return acc_corr

    def combine_vqa(self):   ### combine human and model ! 
        hm = pd.concat([self.model, self.human]) # just the blind conditions 
        hm.pivot_table(  
            index=['model'], 
            values=['correct', 'answer_similarity'], 
            columns=['answer_type'], 
            aggfunc=['mean'] ,
            margins=True,
            margins_name='Average'
        ).to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/1_VQA_human-pretrained-model_similarity-answer_type.tex', float_format="%.1f")   
        
        ### filtered datafrmes 
        df = hm[['meta_model', 'question_id', 'answer_type', 'question_type', 'model', 'question', 'answer', 'output', 'correct', 'answer_similarity']]
        MH_qid = pd.merge( self.model, 
                            self.human.groupby(['question_id', 'answer_type', 'question_type']).mean(numeric_only=True).reset_index() , 
                            on=['question_id', 'answer_type', 'question_type'], 
                            how='left', suffixes=('_model', '_human') )# .dropna(axis=1)
        # len(qids), # model_vqa.groupby(['model', 'question_id']).count()    
        return MH_qid 

    def get_quadrants(pt):
        human_avg_by_qid = human.groupby(['question_type', 'answer_type', 'question_id'])['correct'].mean().reset_index()

        mms=pt.groupby(['question_id', 'question_type', 'answer_type']).mean(numeric_only=True).reset_index() 
        mms = pd.merge(human_avg_by_qid, mms, on =['question_type', 'answer_type', 'question_id'], how='inner', suffixes=('_human', '_model')) 
        # pd.merge(hm, mms, on =['question_id', 'question_type','answer_type'], how='left', suffixes=("", '_avg')).to_csv('./examples/quadrants.csv', encoding="utf-8-sig")   
        
        mms["human_correct"] = (mms["mean_correct"] >= 50).astype(int)  
        mms["model_correct"] = (mms["mean_correct_inst blind"] >= 50).astype(int) 
        mms["cc_quadrant"] = mms.apply(cc_quadrant, axis=1) 

        mms.groupby("cc_quadrant")["MG"].agg(['count', 'mean']).to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/appendix/vqa-quadrant.tex', float_format="%.1f")  
        mms.groupby(["cc_quadrant", 'answer_type', 'new_question_type'])["MG"].agg(['count', 'mean']).to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/appendix/vqa-quadrant-qtype.tex', float_format="%.1f")  
        mms.to_csv('./vqa_quadrants.csv')   
        
        ### BY QUESTION TYPES 
        qt_summary = (
            df[df.dataset == "VQAv2"]
            .groupby(["question_type", "cc_quadrant"])
            .agg(
                MG_mean=("MG", "mean"),
                MG_sem=("MG", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                n=("MG", "count")
            )
            .reset_index()
        ) 
        quad_stat = pd.merge(median_mg_table(mms, "MG"), quadrant_proportions_table(mms), on =['cc_quadrant'], suffixes=('_MG_Acc', ''))
        quad_stat.to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/4_VQA_quad_stat.tex', float_format="%.1f")   
        
        return quad_stat  

class distributional_alignment(): 
    def __init__(): 
        for model in mg.model.unique():  
            df = mg[mg['model'] == model] 
            acc_stat, emb_stat = self.aligned_histogram(df['correct_human'], df['correct_model'], df['answer_similarity_human'], df['answer_similarity_model'], model) 

            alignments = [] 

            alignments.append(acc_stat)
            alignments.append(emb_stat)
        results_df = pd.DataFrame(alignments)
        rename_map = {
            "Spearman_rho": "Spearman_ρ (↑)",
            "Kappa_Quad": "Kappa_Quad (↑)",
            "TVD": "TVD (↓)",
            "JS_Dist": "JS_Dist (↓)",
            "KS_Stat": "KS_Stat (↓)",
            "Wasserstein": "Wasserstein (↓)",
            "Delta_Mean": "Δ_Mean (→0)",  # Bias is best when close to zero
            "metric": "Metric"
        }

        results_df = results_df.rename(columns=rename_map)
        # Debug: Some columns may not be numeric, causing aggfunc='mean' to fail. Select only numeric columns.
        numeric_cols = results_df.select_dtypes(include='number').columns
        # Keep 'model' and 'Metric' for pivot
        pivot_cols = ['model', 'Metric'] + list(numeric_cols.difference(['model', 'Metric']))
        pivot_df = results_df[pivot_cols]
        pivot_df = pivot_df.pivot_table(index='model', columns='Metric', aggfunc='mean')
        pivot_df.to_csv('/home/work/yuna/HPA/evaluation/analysis/tables/alignment.csv')
        pivot_df 
        
    def aligned_histogram(human_acc, model_acc, human_emb, model_emb, modelname='Pretrained VLMs'):  

        x,y,x2,y2= human_acc, model_acc, human_emb, model_emb 
        
        acc_stat = calculate_alignment_suite(x, y) 
        acc_stat['metric'] = 'accuracy' 
        acc_stat['model'] = modelname 

        emb_stat = calculate_alignment_suite(x2, y2) 
        emb_stat['metric'] = 'answer_similarity'
        emb_stat['model'] = modelname 
        
        ### ECDF Plot  
        draw_histograms_vqa(x, y, x2, y2, acc_stat['Pearson_r'], acc_stat['KS_Stat'], emb_stat['Pearson_r'], emb_stat['KS_Stat'], modelname) 

        print(f"Accuracy\n$r={acc_stat['Pearson_r']:.2f}$, KS={acc_stat['KS_Stat']:.2f}")
        print(f"Embedding\n$r={emb_stat['Pearson_r']:.2f}$, KS={emb_stat['KS_Stat']:.2f}") 
        return acc_stat, emb_stat