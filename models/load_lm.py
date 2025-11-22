import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Qwen2_5(): 

    def __init__(self, model_name = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=50, device=1): 
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = f"cuda:{device}" 
        self.max_new_tokens=max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=self.device
        )

    def get_outputs(self, prompt): 

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        # content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("\n")[0] 
        return {"text": response}  ### TODO: HARD CODED as it prints out the history behind   

class internlm3(): 

    def __init__(self, model_dir = "internlm/internlm3-8b-instruct",max_new_tokens=50): 
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        self.model.eval()
        self.max_new_tokens=max_new_tokens 

    def get_outputs(self, prompt): 

        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}, 
        ]
        
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(tokenized_chat, max_new_tokens=self.max_new_tokens, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
        # ]
        prompt = self.tokenizer.batch_decode(tokenized_chat)[0]
        # print(prompt)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        return response 

class internlm2_5:
    
    def __init__(self, model_dir="internlm/internlm2_5-1_8b", max_new_tokens=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens 
        

    def get_outputs(self, prompt):
        
        inputs = self.tokenizer([prompt], return_tensors="pt")
        for k,v in inputs.items():
            inputs[k] = v.cuda()
        gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
        output = self.model.generate(**inputs, **gen_kwargs)
        output = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        
        return output 


class mistral():
    def __init__(self, model_path="mistralai/Mistral-7B-Instruct-v0.2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda") 
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = """{% for message in messages %}
            {{ '<s>' if loop.first else '' }}{{ message['role'].upper() }}: {{ message['content'] }}{% if loop.last %} ASSISTANT:{% endif %}
            {% endfor %}"""
                    
    def get_outputs(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        # ✅ Create attention mask
        attention_mask = inputs.ne(self.tokenizer.pad_token_id).long()

        output_ids = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=model.tokenizer.eos_token_id
        )

        response = model.tokenizer.decode(
            output_ids[0][inputs.shape[-1]:],  # strip prompt
            skip_special_tokens=True
        )
        return response #  self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class Qwen3():
    def __init__(self, model_path="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda") 
        self.max_new_tokens = 50 
                    
    def get_outputs(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)
        return content #  self.tokenizer.decode(outputs[0], skip_special_tokens=True)
