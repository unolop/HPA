import os 
from typing import List, Dict, Any, Iterable 
import json 
import time
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
MODEL = "gpt-4.1-mini-2025-04-14"  # use a snapshot for reproducibility (if available in your account) 

def call_api(system: str, schema: dict[str, Any], name: str, questions: list[str], retries=3) -> list[dict]:
    payload = json.dumps(questions, ensure_ascii=False)
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": payload},
                ],
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                temperature=0,
                seed=42
            )
            data = json.loads(resp.output_text)
            results = data.get("results", [])
            if len(results) != len(questions):
                raise ValueError(f"Length mismatch: got {len(results)}, expected {len(questions)}")
            return results
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  Retry {attempt+1} after {wait}s: {e}")
            time.sleep(wait) 
            
def est_tokens_for_question(q: str) -> int:
    # Conservative heuristic: input tokens
    return max(1, len(q) // 4) + 8

def est_output_tokens_per_item() -> int:
    # Your schema is large; be conservative.
    # Tune this after you run 1-2 batches and observe actual output sizes.
    return 180

def make_batches(items: List[Dict[str, Any]],
                 max_est_total_tokens: int = 18000,
                 hard_max_items: int = 60) -> Iterable[List[Dict[str, Any]]]:
    """
    items: [{"qid":..., "question":...}, ...]
    max_est_total_tokens: combined estimated input+output budget per request
    hard_max_items: absolute cap per batch to avoid huge outputs
    """
    batch = []
    cur = 0
    out_per = est_output_tokens_per_item()

    for it in items:
        q = it["question"]
        add = est_tokens_for_question(q) + out_per

        # start new batch if we'd exceed budget or item cap
        if batch and (cur + add > max_est_total_tokens or len(batch) >= hard_max_items):
            yield batch
            batch = []
            cur = 0

        batch.append(it)
        cur += add

    if batch:
        yield batch
  
