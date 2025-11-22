#!/bin/bash
SAVEDIR='/home/work/yuna/HPA/results/processed/VLM'
MODEL_LIST=(
            "llava-hf/llava-1.5-7b-hf" 
            "llava-hf/llava-v1.6-mistral-7b-hf" 
            "Salesforce/instructblip-vicuna-7b" 

            "google/gemma-3-4b-it"    
            "Qwen/Qwen3-VL-4B-Instruct"
            "Qwen/Qwen3-VL-4B-Thinking"

            "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"  

            "Qwen/Qwen3-VL-8B-Instruct"
            "Qwen/Qwen3-VL-8B-Thinking"

            "llava-hf/llava-1.5-13b-hf" 
            "google/gemma-3-12b-it"  
            ) 

for MODEL in "${MODEL_LIST[@]}"; do  
        python eval_vqa.py --model $MODEL --blind --savedir $SAVEDIR --sample_size 5000 
        # python eval_vqa.py --model $MODEL --dataset Lin-Chen/MMStar # --skip_exist 
        # python eval_vqa.py --model $MODEL --dataset Lin-Chen/MMStar --blind  
    done 