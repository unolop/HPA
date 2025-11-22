# Evaluate with semantic clustering and VQA score
python code/hpa_eval/evaluate.py \
  --human data/human.jsonl \
  --model runs/pred.jsonl \
  --use_semantic_clustering \
  --cluster_thresh 0.8 \
  --embed_model all-MiniLM-L6-v2 \
  --out runs/metrics_semantic.json

# Human–human semantic agreement summary
python analysis/semantic_agreement.py --human data/human.jsonl --embed_model all-MiniLM-L6-v2
