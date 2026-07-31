import json
import shutil
from api.prompts import *
from api.gpt_utils import call_api, make_batches
from dataset.vqav2 import VQADataset_json, VQADataset
from dataset.paths import VQA_IMAGE_DIR, VQA_QUESTIONS, VQA_ANNOT, VQA_1K

BATCH_SIZE = 50 

def get_vqa_questions(dataset='VQA_1K', input_path=None):
    json_path = input_path or VQA_1K
    vqa1k = VQADataset_json(prompt='', \
                            image_dir_path=VQA_IMAGE_DIR, \
                            json_path=json_path)

    dataset = VQADataset( image_dir_path=VQA_IMAGE_DIR,
                question_path=VQA_QUESTIONS,
                annotations_path=VQA_ANNOT,
                prompt='')

    qids = [d['question_id'] for d in vqa1k]  # get the qids from subset
    questions = [q for q in dataset.questions if q['question_id'] in qids]  # get the original questions
    print(f"loaded {len(questions)} questions from VQA validation 1k \ne.g.")
    print(questions[0])

    return questions

def process(questions, mode='CTL', output_path: str = "vqa_extracted.jsonl", batch_size: int = BATCH_SIZE):
    output = []
    batches = make_batches(questions, hard_max_items=batch_size) 

    if mode == 'CTL': 
        system = SYSTEM_CTL
        schema= CTL_SCHEMA 
    else: 
        system = SYSTEM_SEM 
        schema= SEM_SCHEMA  

    with open(output_path, "a", encoding="utf-8") as f:
        for batch in batches:  
            # Extract questions and call API
            questions_text = [q['question'] for q in batch]
            ctl_results = call_api(system, schema, "vqa_ctl_v1", questions_text) 
            
            # Merge results
            current_batch_processed = [
                {**s, **c} for s, c in zip(batch, ctl_results)
            ]
            
            # Write only the NEW items to the file immediately
            for item in current_batch_processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            output.extend(current_batch_processed)
            print(f"Saved {len(output)} total items (Batch size: {len(batch)})")
            
    return output  

def merge_ctl_field(jsonl_path: str, field: str, system: str, schema: dict,
                    batch_size: int = BATCH_SIZE):
    """Add a single new control field to an existing JSONL without re-running the others.

    Records that already have `field` are skipped (safe to resume).
    The original file is backed up as <jsonl_path>.bak before writing.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    pending = [r for r in records if field not in r]
    print(f"{len(pending)} / {len(records)} records need '{field}'")
    if not pending:
        print("Nothing to do.")
        return

    index = {r["question_id"]: r for r in records}

    batches = list(make_batches(pending, hard_max_items=batch_size))
    done = 0
    for batch in batches:
        questions_text = [r["question"] for r in batch]
        results = call_api(system, schema, f"vqa_ctl_{field}", questions_text)
        for record, result in zip(batch, results):
            index[record["question_id"]][field] = result[field]
        done += len(batch)
        print(f"  merged {done}/{len(pending)}")

    shutil.copy(jsonl_path, jsonl_path + ".bak")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written to {jsonl_path}  (backup: {jsonl_path}.bak)")


def main(args):
    mode = args.mode

    if mode == "MERGE_PRONOMINALIZED":
        merge_ctl_field(
            jsonl_path="./dataset/vqa/vqa1k_control.jsonl",
            field="pronominalized",
            system=SYSTEM_CTL_PRONOMINALIZED,
            schema=CTL_PRONOMINALIZED_SCHEMA,
        )
        return

    questions = get_vqa_questions()
    output = process(questions, mode, output_path=f"./dataset/vqa1k_{mode}.jsonl")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["SEM", "CTL", "MERGE_PRONOMINALIZED"],
                        help="SEM: extract semantics; CTL: extract all 4 control types; "
                             "MERGE_PRONOMINALIZED: add pronominalized to existing control JSONL")
    args = parser.parse_args()
    main(args)
