import numpy as np 
import json
import pandas as pd 
from utils.df import *  # result_to_df   aggregate_vqa, model_name_map, calculate_mg   
from utils.corr import interrater_agreement  # get_all_cm, get_agreements, get_finetuned_model_corr, transform_corr_table, plot_unified_agreement_heatmap
from utils.quadrants import * # quadrant_proportions_table, median_mg_table   
from utils.score import get_summary, extract_mc_choice, MODEL_DISPLAY_MAP
from utils.vqa import score_number, score_yes_no, VQAAnswerMapper, vqa_accuracy , PostProcessor 
from utils.plots import scatterplot, get_family, parse_size  
import warnings 

warnings.filterwarnings("ignore") 
import sys 
sys.path.append('..')

    
def test_human_data(human_vqa): 
        
    # with open("/home/work/yuna/HPA/data/training/s1_text/cleaned_n15_text.json", 'r') as f:
    with open("/home/yuna/workspace/HPA/data/training/s1_choice/cleaned_n15_choice.json", 'r') as f:
        data = json.load(f)
    pids = pd.DataFrame(data).participant_id.unique() 
    # pids = np.append(pids, "2e184452_20251205_141511_cleaned")

    trainh = human_vqa[human_vqa['participant_id'].isin(pids)] 
    human_vqa = human_vqa[~human_vqa['participant_id'].isin(pids)] 
    print(len(trainh.participant_id.unique()) , len(human_vqa.participant_id.unique()))  
    
    return trainh, human_vqa 

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

    pcorr['model_type'] = 'vlm'
    pcorr.loc[
        pcorr['model'].isin(['Qwen3 (4B)', 'Qwen3 (8B)']),
        'model_type'
    ] = 'llm' 

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
        self.human = self.human.rename(columns={'pid': 'question_id'}).drop_duplicates(
            ['category', 'l2_category', 'question_id', "participant_id"]) #   = self.clean_df(human_mc)
        self.qids = self.human.question_id.unique()    
        # self.model = self.get_mg(blind_model[blind_model['question_id'].isin(self.qids)]) 
        # self.train_human, self.test_human = test_human_data(self.human) 

        ### MODEL RESULTS 
        model_mc = get_summary('mmstar')
        model_mc = model_mc.rename(columns={'pid': 'question_id'})  
        model_mc = get_training_regime(model_mc)
        model_mc = model_mc.drop_duplicates(subset=['category', 'l2_category', 'model', 'question_id', 'condition', 'family', 'model_size', 'finetuned', 'model_type']) 

        test_model = model_mc[~model_mc['question_id'].isin(self.qids)]
        test_model = self.clean_df(test_model) 
        # self.test_model = test_model[test_model['condition'] == '']
        self.test_model = test_model 
        
        self.model_mc = model_mc[model_mc['question_id'].isin(self.qids) ]     
        # if trained_dataset is empty, set it to Pretrained 
        model_mc['trained_dataset'] = model_mc['trained_dataset'].fillna('Pretrained') 
        self.model_blind = model_mc[(model_mc['condition'] == 'inst blind') & 
                                    (model_mc['trained_dataset'] == 'MMStar') | (model_mc['trained_dataset'] == 'Pretrained')] 
        # self.model_mg = self.get_mg(model_mc, filename='model') 
        # self.test_model_mg = self.get_mg(test_model, filename='test_model_only') 
        # model_mc = self.clean_df(model_mc)

        self.model_list = ['InternVL 3.5 (1B)', 'InternVL 3.5 (2B)', 'InternVL 3.5 (4B)',
                    'InternVL 3.5 (8B)', 'LLaVA-1.5 (7B)', 'LLaVA-Mistral (7B)',
                    'LLaVA-Vicuna (7B)', 'Qwen3 (4B)', 'Qwen3 (8B)', 'Qwen3-VL (2B)',
                    'Qwen3-VL (4B)', 'Qwen3-VL (8B)']
    

    def clean_df(self, df): 
        mmstar_cat_map = {'coarse perception': "CP", 'fine-grained perception': "FP",
                'instance reasoning': "IR", 'logical reasoning': "LR", 'math': "Math",
                'science & technology': 'S&T'} 
        df['category'] = df['category'].map(mmstar_cat_map)
        df['question_id'] = df['question_id'].astype('Int64') 

        return df 

    def get_quadrants(self): 
        mms = pd.merge(
            self.model_blind, 
            self.human.groupby(['question_id', 'l2_category', 'category']).mean(numeric_only=True).reset_index(),
            on=['question_id', 'l2_category', 'category'], 
            how='left', 
            suffixes=("_model", "_human")
        )

        mms["human_correct"] = (mms["correct_human"] >= 50).astype(int)  
        mms["model_correct"] = (mms["correct_model"] >= 50).astype(int) 
        mms["cc_quadrant"] = mms.apply(cc_quadrant, axis=1) 
        mms['dataset'] = "MMStar" 
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
        ) # .to_latex(f'/home/work/yuna/HPA/evaluation/analysis/tables/MMSTAR_MG_{filename}-by-category.tex', float_format="%.1f")
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

    def combine_mh(self, human_acc=None, model_acc=None, metrics='correct'):

        if human_acc is None:
            human_acc = self.human
        if model_acc is None:
            model_acc = self.model

        grouping = ['category', 'l2_category', 'question_id'] # , 'question', 'answer'] 
        # mm = pd.merge(self.model, self.human, on=grouping, how='left', suffixes=('_model', '_human')).groupby().agg({'correct_human': 'mean', 'correct_model': 'mean'}).reset_index()
        # mms = pd.merge(mms, pt.groupby(['category', 'l2_category', 'question_id']).agg({'MG': 'mean'}).reset_index(), on=['category', 'l2_category', 'question_id']) # 'model', , 'correct_model': 'mean'
        # print('combined model and human questions example: ', mm.iloc[0]['question_model'], mm.iloc[0]['question_human']) # .model.unique()   

        human_acc = human_acc.pivot(
                index=grouping, 
                columns="participant_id",
                values=metrics
            )
        model_acc = model_acc.pivot_table( 
            index=grouping,  
            columns="model",
            values=metrics, 
            aggfunc='mean'
        ) 
        answers_pivot = human_acc.join(model_acc, how="inner").reset_index() 

        return answers_pivot    

    def get_corr(self, human_vqa, model_vqa, 
                test_subjects, n_boot = 0 , metrics='correct'): 
        
        human_acc = human_vqa.pivot(
            index="question_id",
            columns="participant_id",
            values=metrics
        )

        res_HH = interrater_agreement(
            human_acc,
            grp1=self.trainh.participant_id.unique(),
            grp2=test_subjects, 
            n_boot=n_boot,
            metric="spearman",
            title=f"Human-Human {metrics}",
        )
        corr = res_HH['corr_mat'].values
        mean_corr = np.nanmean(corr) 
        std_corr  = np.nanstd(corr) 
                
        model_corr = get_finetuned_model_corr(model_vqa, human_vqa, test_subjects)  
        model_corr['family'] = model_corr['model'].map(get_family) 
        model_corr['model_size'] = model_corr['model'].map(parse_size).astype(int)

        # Merge with performance on test set  
        model_correct = self.test_model[self.test_model['condition'] == ''].groupby(
                                    ['model', 'finetuned', 'trained_dataset'])['correct'].mean().reset_index()   
        # model_correct = model_correct[~model_correct['model'].isin(['Qwen3 (4B)', 'Qwen3 (8B)'])] 
        pcorr = pd.merge(model_corr, model_correct, how='left', on=['model', 'finetuned'])   
        
        return mean_corr, std_corr, pcorr  

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
        self.human = self.new_score(self.human).drop_duplicates(subset=['question_id', 'participant_id'])  
        # self.human = self.score_answer_types(self.human, answer_column="answer_normalized", gt_column="visual_gt") 
        self.qids= self.human.question_id.unique().astype(int)
        # self.train_human, self.test_human = test_human_data(self.human) 
        
        # get model intersection subset 
        vqa_1k = get_summary('vqa_1k').drop_duplicates(subset=['condition', 'question_id', 'model'])
        vqa_1k = self.new_score(vqa_1k, 'output', 'multiple_choice_answer')
        vqa_1k = get_training_regime(vqa_1k) 
        
        self.model = vqa_1k[(vqa_1k['question_id'].isin(self.qids)) & (vqa_1k['condition'] == 'inst blind')] # .replace(MODEL_DISPLAY_MAP)
        vqa_5k = get_summary('vqa_5k').drop_duplicates(subset=['condition', 'question_id', 'model', 'strategy', 'blind', 'trained_dataset'])
        vqa_5k = self.new_score(vqa_5k, "output", "multiple_choice_answer") 
        
        # self.model_mg = self.get_mg(vqa_1k[(vqa_1k['question_id'].isin(self.qids))] ) 
        # self.model_mg_sim = self.get_mg(vqa_1k[(vqa_1k['question_id'].isin(self.qids))], 'answer_similarity' ) 
        # self.test_model = get_training_regime(vqa_5k)  
        # self.test_model_mg_acc = self.get_mg(self.test_model, 'correct')   
        # self.test_model_mg_sim = self.get_mg(self.test_model, 'answer_similarity')   
        # self.quadrants = self.get_quadrants() 

    def get_corr(self, human_vqa, model_vqa, metrics='correct'): 
        n_boot = 0 
        test_subjects = self.test_human.participant_id.unique()
        human_acc = human_vqa.pivot(
            index="question_id",
            columns="participant_id",
            values=metrics
        )

        res_HH = interrater_agreement(
            human_acc,
            grp1=self.trainh.participant_id.unique(),
            grp2=test_subjects,
            n_boot=n_boot,
            metric="spearman",
            title=f"Human-Human {metrics}",
        )
        corr = res_HH['corr_mat'].values
        mean_corr = np.nanmean(corr) 
        std_corr  = np.nanstd(corr)
                
        model_corr = get_finetuned_model_corr(model_vqa, human_vqa, test_subjects)  
        model_corr['family'] = model_corr['model'].map(get_family) 
        model_corr['model_size'] = model_corr['model'].map(parse_size).astype(int)
        # model_correct = model_vqa.groupby(['model', 'finetuned'])['correct'].mean().reset_index() 
        model_correct = self.test_human[self.test_human['condition'] == ''].groupby(['model', 'finetuned'])['correct'].mean().reset_index()   
        model_correct = model_correct[~model_correct['model'].isin(['Qwen3 (4B)', 'Qwen3 (8B)'])] 
        pcorr = pd.merge(model_corr, model_correct, how='left', on=['model', 'finetuned'])   
        
        return pcorr, mean_corr, std_corr   

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
                    f"{ex['target_group']}_{i}_model.csv", 
                        encoding="utf-8",
                        index=False
                )

                ex["human"].to_csv(
                    f"/home/work/yuna/HPA/evaluation/analysis/quadrant_examples/"
                    f"{ex['target_group']}_{i}_human.csv",
                        encoding="utf-8",
                        index=False
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
        # print('before' , np.mean(human_vqa['correct']))
        pp = PostProcessor() 
        human_vqa['processed_ans'] = human_vqa[ans_col].apply(pp.postprocess_answer)
        human_vqa['correct'] = human_vqa.apply(
            lambda row: max(
                row['correct'] if pd.notna(row.get('correct')) else 0,
                vqa_accuracy(row["processed_ans"], row[gt_col]) * 100 
            ),
            axis=1
        ) 
        # print('after ', np.mean(human_vqa['correct']))

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
        # human_vqa.to_csv('/home/work/yuna/HPA/evaluation/analysis/human_vqa.csv') 
        return human_vqa 

    def get_mg(self, df, metric='correct', filename='inst-blind_only'): 
        df['new_question_type'] = df["question_type"].apply(map_question_type) 

        pt = df.pivot_table(
            index=['model', 'finetuned', 'answer_type', 'question_type', 'new_question_type', 'question_id'],
            values=metric,
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
        ).to_latex(f'/home/work/yuna/HPA/evaluation/analysis/tables/VQA_MG_{filename}-by-answer_type_{metric}.tex', float_format="%.1f")
        # pt.groupby(['model']).mean(numeric_only=True).to_latex(f'./tables/appendix/mmstar-mg_{filename}.tex', float_format="%.1f")

        return pt.dropna(axis=0, subset=['MG']) # .dropna(axis=1)  

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
