import numpy as np 
import pandas as pd 
import os 
import re 
import json
import sys
from collections import defaultdict
from pathlib import Path


datasets = [ "mmstar","spubench","vqa_1k",  "vqa_5k" ]
conditions = ["_inst_blind", "" ,"_sys_inst_blind", "_blind"]
modelnames = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-2B",
    "OpenGVLab/InternVL3_5-1B",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "OpenGVLab/InternVL3_5-8B",
    "Qwen/Qwen3-VL-8B-Instruct",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "Qwen/Qwen3-VL-4B-Instruct",
    "llava-hf/llava-1.5-7b-hf"]

def get_conditions(path, datasets=datasets, conditions=conditions, modelnames=modelnames):
    filename = os.path.basename(path).replace(".jsonl", "")
    tokens = filename.split("_")
    short_models = {m.split("/")[-1]: m for m in modelnames}

    model_short = None
    model_full = None
    for short, full in short_models.items():
        if short in filename:
            model_short = short
            model_full = full
            break

    dataset = None
    for d in datasets:
        if d in tokens or d in filename:
            dataset = d
            break

    condition = ""
    for c in conditions:
        if c != "" and c in filename:
            condition = c
            break

    return model_full, dataset, condition

def extract_mc_choice(output: str) -> str:
    """
    Extract the predicted answer (A, B, C, D, etc.) from model output.
    Uses multiple heuristics to find the answer.
    """
    if not output:
        return ""
    
    output = output.strip()
    
    # Pattern 1: Look for explicit answer statements
    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*([A-D])",
        r"(?:the\s+)?(?:correct\s+)?answer[:\s]*([A-D])",
        r"(?:option\s+)?([A-D])\s+is\s+(?:the\s+)?correct",
        r"(?:I\s+)?(?:would\s+)?choose\s+(?:option\s+)?([A-D])",
        r"(?:I\s+)?(?:would\s+)?select\s+(?:option\s+)?([A-D])",
        r"^([A-D])(?:[:\.\)]|\s|$)",  # Answer at the start
        r"\n([A-D])(?:[:\.\)]|\s|$)",  # Answer after newline
        r"(?:Therefore|Thus|So|Hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(?:option\s+)?([A-D])",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Pattern 2: Look for the last standalone letter A-D
    matches = re.findall(r'\b([A-D])\b', output)
    if matches:
        return matches[-1].upper()
    
    # Pattern 3: Check if output is just a single letter
    if len(output) == 1 and output.upper() in 'ABCD':
        return output.upper()
    
    return ""

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
    
def split_thinking(s):
    if '</think>' in s:
        splits = s.split('</think>')
        prediction = splits[-1].strip()
        if len(splits) == 2 and '<think>' in splits[0]:
            thinking = splits[0].split('<think>')[1].strip()
        else:
            thinking = '</think>'.join(splits[:-1])
            thinking += '</think>'
            warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
    else:
        thinking = ''
        prediction = s
    return (prediction, thinking)

contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }

numbers = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

punctuations = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

class PostProcessor: 
    
    def __init__(self):
        self.contractions = contractions
        self.manualMap = numbers 
        self.articles = ["a", "an", "the"]
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = punctuations 

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        # print(inText, outText)
 
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        # print(inText, 'out:', outText)
        return outText

    def postprocess_answer(self, answer):
        
        if isinstance(answer, str):
            answer = answer.split("\n")[-1].split("\t")[-1]
            # generation = generation.replace("\n", " ").replace("\t", " ").strip()

        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0]
    
        if "[INST]" in answer and "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        
        if "inst" in answer:
            answer = answer.split("inst")[-1].strip()
            
        if "USER" in answer and "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()
        
        if "assistant" in answer:
            answer = answer.split('assistant')[-1].strip()
        
        if "Answer:" in answer: 
            answer = answer.split("Answer: ")[-1]

        if "answer:" in answer:  ## FOR LLAVA 1.5 
            answer = answer.split("answer: ")[-1]
        # else:
        #     answer = answer.strip().split()[-1] if isinstance(answer, str) else ""

        # Remove extraneous tokens
        for token in [
            '<s>', '</s>', '<|im_end|>', '[/INST]', '<|im_start|>', 
            'ASSISTANT:', 'assistant\n', 'inst'
        ]:
            answer = answer.replace(token, '')

        # answer = answer.strip()
        if not answer:
            answer = ""

        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        
        return answer
