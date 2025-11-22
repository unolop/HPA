import pandas as pd
import json
import time
import argparse 
from openai import OpenAI
from api import gpt_api

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate(question, options):

    prompt = f"""
            You are a precise translation and data-formatting assistant.

            Task:
            Given a multiple-choice question and its options in English, 
            return the result as a strict JSON dictionary with the following fields:
            - question_kr: Translate the QUESTION into Korean 
            - options: Insert translated korean options after its English choices in the same string format. 

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

def main(args): 
    output_file = os.path.join(args.output_dir, os.path.basename(args.question_file) )
    # Load CSV 
    df = pd.read_csv(args.question_file)
    df["question_kr"] = None
    df["options"] = None

    for idx, row in df.iterrows():
        parsed = translate_row(row)
        if parsed:
            df.at[idx, "question_kr"] = parsed["question_kr"]
            df.at[idx, "options"] = parsed["options"] # , ensure_ascii=False)
        time.sleep(0.2)  # avoid rate limits

    df.to_csv(output_file, index=False)
    print(f"Done!  saved in {output_file}")

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_file', type=str) 
    parser.add_argument('--output_dir', type=str, default="./processed") 
    main(client, args)