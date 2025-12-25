import json
import seaborn as sns
import pandas as pd 
from utils.df import *  # result_to_df   aggregate_vqa, model_name_map, calculate_mg   
from utils.corr import get_all_cm, get_agreements, transform_corr_table, plot_unified_agreement_heatmap
from utils.quadrants import * # quadrant_proportions_table, median_mg_table   
from utils.score import get_summary, VQAAnswerMapper, extract_mc_choice
from utils.ac1 import interrater_agreement_ac1 
from utils.plots import scatterplot, get_family, parse_size  
import sys 
sys.path.append('..')


pretrained_models  = [
    'InternVL 3.5 (1B)', 'InternVL 3.5 (2B)', 'InternVL 3.5 (4B)',
    'InternVL 3.5 (8B)', 'LLaVA-v1.5 (7B)', 'LLaVA-v1.6-Mistral (7B)',
    'LLaVA-v1.6-Vicuna (7B)','Qwen3-VL (2B)', 'Qwen3-VL (4B)', 'Qwen3-VL (8B)', 
    'Qwen3 (4B)', 'Qwen3 (8B)', ] 

class SpubenchResults(): 
    def __init__(self):
        
        df = get_summary(dataset='spubench')
        df['spurious_correlation_type'] = df['spurious_correlation_type'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df= df.explode('spurious_correlation_type') 
        spu= pd.read_json("/home/work/yuna/HPA/dataset/annotation.json").reset_index()
        spu['pid'] = spu['index']
        df = pd.merge(df, spu, on="pid", how="left", suffixes=("", "_annot"))
        df = df.loc[:, ~df.columns.duplicated()]
        df["mc_choice"] = df["output"].apply(extract_mc_choice)
        df["answer"] = df["answer"].fillna(df["answer_annot"])
        df["correct"] = (
            df["mc_choice"].notna()
            & df["answer"].notna()
            & (df["mc_choice"].str.strip().str[0].str.upper()
            == df["answer"].str.upper())
        )
        df = df.dropna(axis=1, how="all")
        df.groupby(['model'])['correct'].mean().reset_index().to_csv('./sputbench.csv', encoding="utf-8-sig")
        df['model'] = df['model'].map(model_name_map)
        df = df.drop_duplicates(subset=['pid', 'model'])
        df = df[df['condition'] == '']
        df = df.explode("spurious_correlation_type_annot")
        df['finetuned'] = df['model'].str.split('|').str[1]
        df['model'] = df['model'].str.split('|').str[0].str.strip()
        df['finetuned'] = df['finetuned'].fillna('baseline')
        df.pivot_table(index=['model'], 
                                            columns='spurious_correlation_type_annot', 
                                            values='correct').to_csv('./spubench-corr.csv', encoding="utf-8-sig")   

        df['bias'] = df['spurious_correlation_type_annot'].map({
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

    def finetuning_effect(self): 
        bs = df[df["finetuned"] == 'baseline']
        ft = df[df["finetuned"] != 'baseline']

        df = pd.merge(bs,ft, on=['model', 'pid', "bias"], how='right', suffixes=("_baseline", "")).dropna(axis=1) 
        df['delta'] = df['correct_baseline'].astype(float) - df['correct'].astype(float)
        df=df[['model', 'finetuned', 'pid', 'bias', 'delta']]
        pt = (
            df.groupby(['model', 'finetuned'])['delta']
            .mean()
            .round(3)
        ) 
        pt.to_latex(
            './tables/4_spubench-corr-finetuned-delta.tex',
            float_format="%.3f",
            multirow=True
        )
        pt = df.pivot_table(
            index=['model', 'finetuned',], 
            columns=['bias'] , 
            values=['delta']).dropna(axis=1) 
        pt.reset_index(level=0, drop=True)
        pt.columns = pt.columns.droplevel(0)

        pt.to_latex('./tables/appendix/4_spubench-corr-finetuned.tex', float_format="%.3f")        
        pt= df.pivot_table(index=['model'], 
                    columns='spurious_correlation_type_annot', 
                    values='correct') # , aggfunc=['mean'] 

        plt.figure(figsize=(10, 10))
        sns.heatmap(
            pt,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0, vmax=1,
            linewidths=0.5
        )
        plt.ylabel("Model")
        plt.xlabel("Spurious correlation type")
        plt.title("Accuracy by model and spurious correlation type")
        plt.tight_layout()
        plt.show()

class MMStarResults(): 
    def __init__(self, results_path='/home/work/yuna/HPA/evaluation/scored/humans/human_mc_per_question.json', models='all'): 

        with open(results_path) as f:
            raw_data = json.load(f) 

        human_mc = pd.DataFrame(raw_data).rename(columns={'\ufeffquestion_num': 'question_num'}) 
        human_mc['model'] = 'Humans'
        human_mc['condition'] = 'inst blind' 
        human_mc = self.clean_df(human_mc)

        self.qids = human_mc.question_id.unique()    
        self.human = human_mc[human_mc['participant_id']!= '13f54aa2_20251204_125847'] 
        self.human['answer_similarity'] = self.human['answer_similarity'] * 100 
        self.human['correct'] = self.human['correct'] * 100  
        
        ### MODEL RESULTS 
        model_mc = get_summary('mmstar')
        model_mc = self.clean_df(model_mc)

        # self.model = self.get_mg(blind_model[blind_model['question_id'].isin(self.qids)]) 
        self.model_test = self.clean_df(model_mc[~model_mc['question_id'].isin(self.qids)] )
        self.model_test_mg = self.get_mg(self.model_test, filename='test_model_only') 
        self.model = model_mc[model_mc['condition'] == 'inst blind'] 
        self.model = model_mc[model_mc['condition'] == 'inst blind'] 

        self.human_list = self.human.participant_id.unique() 
        self.model_list = self.model.model.unique() 

        if models == 'pretrained':
            self.model_list = [m for m in self.model_list if 'SFT' not in m and 'JS' not in m]
        elif models == 'all':
            pass 
        else: 
            self.model_list = [m for m in self.model_list if models in m] # by model family 

    def clean_df(self, df): 
        if "question_id" not in df.columns: 
            df = df.rename(columns={'pid': 'question_id'})
        try: 
            df['question_id'] = df['question_id'].astype('Int64') 

        except Exception as e: 
            print(e, df.head())
            breakpoint() 
        return df.drop_duplicates(subset=['model', 'question_id', 'condition'])  

    def get_mg(self, df, filename='inst-blind_only'): 
        pt = df.pivot_table(
            index=['model', 'category', 'l2_category', 'question_id'],
            values='correct',
            columns='condition',
            aggfunc='mean'
        ).reset_index()

        pt['MG'] = pt[''] - pt['inst blind']
        pt['Delta_Inst'] = pt['blind'] - pt['inst blind']
        pt[pt['question_id'].isin(self.qids)].pivot_table(
            index=['model'],
            values='MG',
            columns=['category', 'l2_category'],
            aggfunc='mean'
        ).to_latex(f'/home/work/yuna/HPA/evaluation/analysis/tables/MMSTAR_MG_{filename}-by-category.tex', float_format="%.1f")
        # pt.groupby(['model']).mean(numeric_only=True).to_latex(f'./tables/appendix/mmstar-mg_{filename}.tex', float_format="%.1f")

        return pt 

    def get_tables(self): 
        pd.concat([self.model, self.human]).pivot_table(    
                    index=['model'], 
                    values=['correct'], 
                    columns=['category'], 
                    aggfunc=['mean'], # ,'count' 
                    margins=True,
                    margins_name='Total Average'
                ).to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/MMStar_blind_by-category.tex', float_format="%.1f")   

        pd.concat([self.model, self.human]).pivot_table(    
                    index=['model'], 
                    values=['correct', 'MG'] , 
                    columns=['condition'], 
                    aggfunc=['mean'], # ,'count' 
                    margins=True,
                    margins_name='Total Average'
                ).to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/MMStar_blind_by-conditions.tex', float_format="%.1f")   

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
            corr_mat=cc['corr_mat'].T.reset_index().melt(id_vars=['index'], var_name='subj', value_name='corr')    
            corr_mat['category'] = cc['category']
            model_corr.append(corr_mat) 
        model_corr= pd.concat(model_corr).dropna()

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
        self.human = self.clean_df(human_vqa) 
        self.qids= human_vqa.question_id.unique().astype(int)
        
        # get model intersection subset 
        vqa_1k = get_summary('vqa_1k').drop_duplicates(subset=['condition', 'question_id', 'model'])
        self.model = vqa_1k[(vqa_1k['question_id'].isin(self.qids)) & (vqa_1k['condition'] == 'inst blind')] # .replace(MODEL_DISPLAY_MAP)
        self.model_mg = self.get_mg(vqa_1k[(vqa_1k['question_id'].isin(self.qids))], 'subset')  
        self.test_model = get_summary('vqa_5k').drop_duplicates(subset=['condition', 'question_id', 'model'])
        self.test_model_mg = self.get_mg(self.test_model, '5k-test')   
    
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

        human_vqa.to_csv('/home/work/yuna/HPA/evaluation/analysis/human_vqa.csv') 
        return human_vqa 

    def get_mg(self, df, filename='inst-blind_only'): 
        df['new_question_type'] = df["question_type"].apply(map_question_type) 

        pt = df.pivot_table(
            index=['model', 'answer_type', 'question_type', 'new_question_type', 'question_id'],
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

            ### PLOT SCATTER  
            pcorr = self.get_vqa_corr()  # pcorr = pd.merge(acc_corr, sim_corr, on =['model']) 
            df = calculate_mg(self.test_model, filename='vqa_5k').groupby(['model']).mean(numeric_only=True).reset_index()  
            pcorr = pd.merge(pcorr, df, on=['model']) 
            
            pcorr = pcorr[['model', 'acc_corr', 'sim_corr', 'MG_Acc', 'MG_S']]
            pcorr['family'] = pcorr['model'].map(get_family)
            pcorr['model_size'] = pcorr['model'].map(parse_size)  
            scatterplot(pcorr, x_col='acc_corr', y_col='MG_Acc', title='accuracy') 
            scatterplot(pcorr, x_col='sim_corr', y_col='MG_S', title='simliarity')   

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
        
        mms["human_correct"] = (mms["mean_correct"] >= 50).astype(int)  
        mms["model_correct"] = (mms["mean_correct_inst blind"] >= 50).astype(int) 
        mms["cc_quadrant"] = mms.apply(cc_quadrant, axis=1) 

        mms.groupby("cc_quadrant")["MG"].agg(['count', 'mean']).to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/appendix/vqa-quadrant.tex', float_format="%.1f")  
        mms.groupby(["cc_quadrant", 'answer_type', 'new_question_type'])["MG"].agg(['count', 'mean']).to_latex('/home/work/yuna/HPA/evaluation/analysis/tables/appendix/vqa-quadrant-qtype.tex', float_format="%.1f")  
        mms.to_csv('./vqa_quadrants.csv')   
        
        # mms["human_correct"] = (human_avg_by_qid["correct"] >= 50).astype(int)  
        # mms["model_correct"] = (mms["mean_correct_inst blind"] >= 50).astype(int) 
        # pd.merge(hm, mms, on =['question_id', 'question_type','answer_type'], how='left', suffixes=("", '_avg')).to_csv('./examples/quadrants.csv', encoding="utf-8-sig")   
        
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