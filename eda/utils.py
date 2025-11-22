from glob import glob 
import pandas as pd 
import os 
import json 
from typing import List, Any
import os

def translate_question(client, question):
    prompt = f"""
            You are a precise translation and data-formatting assistant.

            Task:
            Given a multiple-choice question and its options in English, 
            return the result as a strict JSON dictionary with the following fields:
            - question_kr: Translate the QUESTION into Korean 
            
            Input:
            QUESTION: {question}
            """

    response = client.chat.completions.create(
        model="gpt-5-nano",   # or your preferred model
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    )

    output = response.choices[0].message.content.strip()
    return output

def translate_mmstar(client, question, options):
    prompt = f"""
            You are a precise translation and data-formatting assistant.

            Task:
            Given a multiple-choice question and its options in English, 
            return the result as a strict JSON dictionary with the following fields:
            - question_kr: Translate the QUESTION into Korean 
            - options: Insert translated korean options after its English choices separated by a blank space in the same string format. Do not remove the original English options. Numbers don't need to be translated in both languages.

            Input:
            QUESTION: {question}
            OPTIONS_STRING: {options}
            """

    response = client.chat.completions.create(
        model="gpt-5-nano",   # or your preferred model
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    )

    output = response.choices[0].message.content.strip()
    return output

def contains_korean(text):
    pattern = re.compile(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]')
    return bool(pattern.search(str(text)))
    
def convert_results(input_csv, groupname): 
    output_folder = f"/home/work/yuna/HPA/results/{groupname}"
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(input_csv))[0] + '.jsonl'
                ) 
    
    if filename in os.listdir(output_folder): 
        return 

    with open(input_csv, "r", encoding="utf-8") as fin, open(filename, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            if 'vqa' in input_csv: 
                item = process_vqa(row) 
            else: 
                item = row 
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved JSONL to: {filename}")
    
def answer_similarity(answer, gt:list): 
    avg = [] 
    # print(gt, answer)

    for gta in gt : 
        gta = gta['answer'] 
        embeddings = encoder.encode([answer, gta]) 
        # print(embeddings.shape) # [3, 384]
        similarities = encoder.similarity(embeddings, embeddings)
        avg.append(similarities[1,0]) 
        # print(similarities) 

    return np.mean(avg) 

def get_annotations(): 
    annotations_path = "/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json"

    ans = {}  
    with open(os.path.join(annotations_path), 'r') as f: 
        annotations = json.load(f)['annotations'] 
    adf = pd.DataFrame(annotations)
    adf['question_id'] = adf['question_id'].astype(str) 
    
    return adf 

def read_human_results(folder='./responses'): 
    dfs=[]
    print('Reading all the excel files in responses folder')
    for f in glob(f'{folder}/*.xlsx'): 
        dfs.append(pd.read_excel(f)) 
    df = pd.concat(dfs)
    df = df.melt(id_vars=['Timestamp', "이름 또는 이니셜 Name or Initials ", 'Score']).rename(columns={'variable': 'question', 'value': 'answer', "이름 또는 이니셜 Name or Initials ": "subj"})
    # df['question'] = df['question'].str.split('\n').str[0]
    df[['question', 'question (Korean)']] = df['question'].str.split('\n', expand=True)[[0,1]] 
    df = df.drop(columns=['Score']).dropna(axis=0)
    
    return df 


annotations_path = "/home/work/yuna/VLMEval/data/v2_mscoco_val2014_annotations.json"
with open(os.path.join(annotations_path), 'r') as f: 
    annotations = json.load(f)['annotations'] 

def get_answer(question_id: Any) -> List[str]:
    
    df = pd.DataFrame(annotations)
    df['question_id'] = df['question_id'].astype(int) 
    target_row = df[df['question_id'] == question_id]
    return target_row.iloc[0]['answers']

missing = [
    {"question": 'Is there anyone on the photo?',
     "question_id": 181499007},
    {"question": 'Does this photo show carrots in a basket?',
     "question_id": 143671000},
    {"question": 'Is there a yield sign in the photo?',
     "question_id": 376493000},
    {"question": 'Is the animal in the photo one of the "big 5" game animals?',
     "question_id": 502630003},
    {"question": 'Is this a photo in Africa?',
     "question_id": 526143000},
    {"question": 'Is there a baby elephant in this photo?',
     "question_id": 16704005},
    {"question": 'What sport is being played in this photo?',
     "question_id": 409964009},
    {"question": 'What type of building can this photo be found in?',
     "question_id": 294992003} 
    ] 
# allqs = pd.concat([allqs, pd.DataFrame(missing)])

with open(os.path.join("/home/work/yuna/VLMEval/data/v2_OpenEnded_mscoco_val2014_questions.json"), 'r') as f: 
    annotations = json.load(f)
def get_question_id(df): 
    allqs = annotations['questions'] 
    allqs = pd.DataFrame(allqs)
    allqs['question_id'] = allqs['question_id'].astype('int') 
    allqs = allqs.drop_duplicates(subset=['question'], keep='first') 

    return pd.merge(df, allqs, on=['question'], how='left') 


def load_data_to_dataframe(file_path):
    """
    Reads a CSV or XLSX file into a pandas DataFrame.

    Args:
        file_path (str): The full path to the CSV or XLSX file.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame, or None if the file 
                                  format is unsupported or an error occurs.
    """
    # Get the file extension and convert it to lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    try:
        if file_extension == '.csv':
            # Use read_csv for CSV files
            print(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            
        elif file_extension in ['.xlsx', '.xls']:
            # Use read_excel for Excel files
            print(f"Reading Excel file: {file_path}")
            df = pd.read_excel(file_path)
            
        else:
            print(f"Error: Unsupported file format '{file_extension}'. Please use .csv or .xlsx.")
            return None

        print(f"\nSuccessfully loaded DataFrame (Shape: {df.shape})")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None