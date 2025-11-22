from evalscope import run_task, TaskConfig
from models.swift import SwiftVLMEngine
from custom_dataset import CustomDataset  # Import the custom dataset

def main(args):
    
    custom_model = SwiftVLMEngine(
        model_name=args.model,
        model_args={"revision": "master", "precision": "torch.float16", "device_map": "auto", 
                    "do_sample": True,
                    "top_k": 50,
                    "logprobs": True,
                    "top_logprobs": 5,
                    "seed": 42
                    }
    )

    # Configure the evaluation task
    task_config = TaskConfig(
        model=custom_model,
        datasets=[args.dataset],
        limit=args.limit
    )

    # Run the evaluation
    results = run_task(task_cfg=task_config)

if __name__=="__main__":
    import argparse 
    
    parser = argparse.ArgumentParser(description="VLM Benchmarks on evalscope with MS-SWIFT models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Models from SWIFT/HuggingFace") 
    parser.add_argument("--dataset", type=str, default="mm_star", help="Benchmark dataset") 
    parser.add_argument("--limit", type=int, default=None, help="Limit of dataset") 
    
    args = parser.parse_args() 
    main(args)    