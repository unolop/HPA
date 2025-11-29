import os 
import json 

def find_id_key(existing_keys): 
    current_item_id = None
    id_keys = ['idx', 'qid', 'question_id', 'index']
    
    for key in id_keys:
        if key in existing_keys:
            current_item_id = key
            break

    create_new_key = False 
    if current_item_id is None : 
        current_item_id = 'pid'

    return current_item_id 

def skip_processed_idx(current_item_id, output_jsonl_path): 

    processed_ids = set()
    if os.path.exists(output_jsonl_path):
        print(f"'{output_jsonl_path}' exists.")
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        # Assuming the ID field is called 'qid'
                        processed_ids.add(item[current_item_id])
                    except json.JSONDecodeError:
                        print(f'detected malformed {item}')
                        continue  # Skip malformed lines
        print(f"총 {len(processed_ids)}개의 항목을 건너뜁니다.")
    return processed_ids
