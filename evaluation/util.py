import os 
import json 

### For inference 
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
