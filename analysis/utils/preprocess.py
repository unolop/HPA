import os 
import re 
import json
import random 

DATASETS = ["mmstar", "spubench", "vqa_1k", "vqa_5k", 'okvqa', 'textvqa']
CONDITIONS = ["_inst_blind", "", "_sys_inst_blind", "_blind"]

MODELNAMES = [
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-2B",
    "OpenGVLab/InternVL3_5-1B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-1.5-7b-hf",
    "Qwen3-4B-Base", 
    "Qwen3-0.6B-Base", 
    "Qwen3-8B-Base", 
    "Qwen3-4B", 
    "Qwen3-0.6B", 
    "Qwen3-8B", 
]

DATASET_TYPE = {
    'mmstar': 'multi-choice',
    'spubench': 'multi-choice',
    'vqa_1k': 'vqav2',
    'vqa_5k': 'vqav2',
    'textvqa': 'open-ended', 
    'okvqa': 'open-ended',  
}

def read_and_clean_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                
                # *** CLEANING STEP FOR 'qid' ***
                # Convert the 'qid' field to a string, regardless of its original type (int, float, or string).
                if 'qid' in d:
                    d['qid'] = str(d['qid'])
                
                data.append(d)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line in {path}: {e}\nLine: {line.strip()}")
                # You might want to skip the bad line or raise the error here
                continue
    return data

def mix_original(original, new_data, combined_output_path ): 
        ### Mixed dataset 
        data1 = read_and_clean_jsonl(original)
        data2 = read_and_clean_jsonl(new_data)
        combined = data1 + data2
        random.shuffle(combined)
        # Do something with shuffled combined
        print(f"Loaded {len(data1)} + {len(data2)} = {len(combined)} examples. First example:")
        # Save the combined shuffled data to a new JSONL file
        with open(combined_output_path, "w", encoding="utf-8") as fout:
            for ex in combined:
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Combined dataset saved to {combined_output_path}")
    # np.random.shuffle(examples) 

def get_conditions(path, datasets=DATASETS, conditions=CONDITIONS, modelnames=MODELNAMES):
    """Extract model, dataset, condition from filename."""
    filename = os.path.basename(path).replace(".jsonl", "")

    tokens = filename.split("_")
    short_models = {m.split("/")[-1]: m for m in modelnames}

    model_full = None
    for short, full in short_models.items():
        if short in path:
            model_full = full
            break
    if 'finetuned' in path : 
        trained = path.split('/')[-2] 
        model_full += f"/{trained}" 

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
        
        # 숫자 단어 -> 숫자 매핑 추가
        self.word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
            'once': '1', 'twice': '2', 'single': '1', 'double': '2', 'triple': '3',
        }

    def extract_number(self, text):
        """텍스트에서 숫자 추출"""
        text_lower = text.lower()
        
        # 1. 숫자 단어가 있으면 변환
        for word, num in self.word_to_num.items():
            if word in text_lower.split():
                return num
        
        # 2. 숫자가 직접 있으면 추출
        numbers_found = re.findall(r'\d+', text)
        if numbers_found:
            return numbers_found[0]
        
        return text 


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

        if "answer:" in answer:
            answer = answer.split("answer: ")[-1]

        for token in [
            '<s>', '</s>', '<|im_end|>', '[/INST]', '<|im_start|>', 
            'ASSISTANT:', 'assistant\n', 'inst'
        ]:
            answer = answer.replace(token, '')

        if not answer:
            answer = ""

        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        answer = self.extract_number(answer)
        
        return answer.strip()