import os
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df

class CustomDataset(CustomVQADataset):
    """Inherit from CustomVQADataset instead of defining standalone class"""
    
    def __init_subclass__(cls, **kwargs):
        """This runs when the class is defined, not instantiated"""
        super().__init_subclass__(**kwargs)
        print("✅ CustomDataset class defined")

    def load_data(self, dataset):
        # Load custom dataset
        print('loaded dataset', dataset)
        data_path = os.path.join("/home/work/yuna/HPA/data/vqav2_1k", f'{dataset}.tsv')

        return load(data_path)
        
    def build_prompt(self, line):
        msgs = super().build_prompt(line)  # Call parent class method
        # Add prompts or custom instructions here
        msgs[-1]['value'] += '\nAnswer the question in one word or phrase.'
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        print(data)
        
        # ========Compute the evaluation metric as needed=========
        # Exact match
        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)
        # Save the result
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret
        # ========================================================

# Register your custom dataset class
# This tells VLMEvalKit to use CustomDataset when 'custom_vqa' is referenced
        
# Keep the following code and override the default dataset class
# Create a model instance

CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate
print("✅ Monkey-patch applied to CustomVQADataset")