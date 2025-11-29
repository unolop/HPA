import json 
from openai import OpenAI
from sklearn.model_selection import train_test_split 
from utils import translate #,translate_question
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY) # os.getenv("OPENAI_API_KEY"))

# Sample new questions 
def sample_new_qs(df, num_sample=0, types=['category', 'l2_category'], answer_type='choice', preprocess=True, filename=None):  

    # preprocessing 
    df = df.rename(columns={"Unnamed: 0": 'qid', 'question_id': 'qid', 'question': 'question_en'}).dropna(axis=0) 
    df['qid'] = df['qid'].astype('Int64') 
    df['answer_type'] = answer_type 

    ## split options from questions 
    if answer_type == 'choice' and 'options' not in df.columns:     
        df[['question_en', 'options']] = df['question_en'].str.split('Options:', n=1, expand=True)

    if num_sample > 0 : ### sample new quetsions 
        print(f"sampling new #{num_sample} from {types}")

        df, _ = train_test_split(
            df,
            train_size=num_sample, 
            # stratify=df[types], 
            random_state=42
        )  

    if preprocess: 

        df['question_kr'] = None 

        for idx, row in df.iterrows():
            question = row["question_en"]
            if 'options' in row.keys(): 
                options = row["options"]
                output = translate(client, question, options) 
            else: 
                output = translate_question(client, question) 
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                print("JSON decode error:", output)
                parsed = None
            print(parsed)
            df.at[idx, 'question_kr'] = parsed['question_kr']
            if 'options' in row.keys(): 
                df.at[idx, 'options'] = parsed['options']

    if filename is not None: 
        df.to_csv(f'./questions/processed/{filename}_{answer_type}.csv')
        print(f'saved to ./questions/processed/{filename}_{answer_type}.csv') 
    
    # print(len(v2.columns),sorted( v2.columns))

    return df 

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
