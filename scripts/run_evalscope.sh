# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-VL-4B-Instruct # --limit 10 
# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-VL-4B-Thinking # --limit 10 
# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-VL-2B-Instruct # --limit 10 
# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-VL-2B-Thinking # --limit 10 

# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model OpenGVLab/InternVL3_5-4B # --limit 10 
# CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model OpenGVLab/InternVL3_5-2B # --limit 10 

CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-1.7B --data vqav2_1k/val
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-0.6B --data vqav2_1k/val
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-4B --data vqav2_1k/val
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-14B --data vqav2_1k/val
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --model Qwen/Qwen3-8B --data vqav2_1k/val