# =============================================================================
# Blind VQA Ablation Study - Experiment Configuration
# =============================================================================
#
# Research Question: 
# How do VLMs rely on linguistic priors vs. visual information in VQA?
#
# Approach:
# 1. Collect human "blind" VQA responses (no images, confidence ratings)
# 2. Train models to match human blind performance
# 3. Compare with models that see images
# 4. Analyze what linguistic priors are captured
#
# =============================================================================

experiment:
  name: "blind_vqa_linguistic_priors"
  version: "1.0"
  description: "Study of linguistic priors in VQA through blind human annotation"

# =============================================================================
# Models
# =============================================================================
models:
  # Primary model for full ablation study
  primary: "OpenGVLab/InternVL3_5-2B"
  
  # Additional models for scale comparison
  scale_comparison:
    - "OpenGVLab/InternVL3_5-1B"
    - "OpenGVLab/InternVL3_5-2B"  
    - "OpenGVLab/InternVL3_5-4B"
    - "OpenGVLab/InternVL3_5-8B"
  
  # Why InternVL3.5?
  # - Consistent architecture across scales
  # - Good baseline performance on VQA
  # - Well-supported by SWIFT
  # - Open weights

# =============================================================================
# Benchmarks
# =============================================================================
benchmarks:
  vqav2:
    description: "VQAv2 validation set (1k subsample)"
    questions_file: "v2_OpenEnded_mscoco_val2014_questions.json"
    annotations_file: "v2_mscoco_val2014_annotations.json"
    images_dir: "coco_val2014"
    human_questions_per_benchmark: 300
    total_val_questions: 5000  # For evaluation
    
  mmstar:
    description: "MMStar benchmark"
    questions_file: "mmstar_questions.json"
    annotations_file: "mmstar_annotations.json"
    human_questions_per_benchmark: 300
    
  mmspubench:
    description: "MMSPUBench - spurious correlation benchmark"
    questions_file: "mmspubench_questions.json"  
    annotations_file: "mmspubench_annotations.json"
    human_questions_per_benchmark: 300

# =============================================================================
# Ablation Conditions
# =============================================================================
ablations:
  A0:
    name: "ZeroShot"
    description: "Zero-shot baseline - no training"
    training_data: null
    image_type: null
    loss: null
    purpose: "Measure pretrained model's linguistic priors"
    
  A1:
    name: "SFT_GT"
    description: "Standard SFT with ground truth + real images"
    training_data: "ground_truth"
    image_type: "real"
    loss: "cross_entropy"
    purpose: "Upper bound - standard fine-tuning"
    
  A2:
    name: "SFT_GT_Blind"
    description: "SFT with ground truth + black images"
    training_data: "ground_truth"
    image_type: "black"
    loss: "cross_entropy"
    purpose: "What can model learn from GT without images?"
    
  A3:
    name: "SFT_Human_Blind"
    description: "SFT with human answers + black images (no confidence)"
    training_data: "human"
    image_type: "black"
    loss: "cross_entropy"
    purpose: "Match human linguistic priors, equal weighting"
    
  A4:
    name: "Soft_Human_Blind"
    description: "Soft SFT with confidence weighting"
    training_data: "human"
    image_type: "black"
    loss: "confidence_weighted_ce"
    confidence_weighting: true
    label_smoothing: true
    purpose: "Match human priors with uncertainty modeling"
    
  A5:
    name: "Soft_Human_Blind_KL"
    description: "Soft SFT + KL regularization"
    training_data: "human"
    image_type: "black"
    loss: "confidence_weighted_ce + kl"
    confidence_weighting: true
    label_smoothing: true
    kl_regularization: true
    kl_weight: 0.1
    purpose: "RECOMMENDED - Match priors while preserving visual capability"

# =============================================================================
# Human Data Collection
# =============================================================================
human_data:
  participants_per_question: 10-30
  questions_per_benchmark: 300
  total_questions: 900  # 3 benchmarks × 300
  
  csv_format:
    columns:
      - question_num
      - qid
      - answer
      - confidence  # 1-5 scale
      - time_spent_seconds
      - answer_timestamp
    
  confidence_scale:
    1: "Very uncertain - guessing"
    2: "Somewhat uncertain"
    3: "Neutral"
    4: "Somewhat confident"
    5: "Very confident"

# =============================================================================
# Training Configuration
# =============================================================================
training:
  # Shared across all ablations
  learning_rate: 2e-5
  num_epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 16
  warmup_ratio: 0.05
  weight_decay: 0.1
  lr_scheduler: "cosine"
  
  # LoRA configuration
  lora:
    rank: 32
    alpha: 64
    dropout: 0.05
    target_modules: "all-linear"
  
  # Freezing strategy
  freeze:
    llm: false
    vit: true  # Always freeze for blind training
    aligner: true
  
  # Soft supervision (A4, A5)
  soft_supervision:
    weight_strategy: "linear"  # linear, quadratic, inverse, softmax
    confidence_min_weight: 0.2
    confidence_max_weight: 1.0
    use_label_smoothing: true
    min_smoothing: 0.0  # For high confidence
    max_smoothing: 0.2  # For low confidence
  
  # KL regularization (A5)
  kl_regularization:
    weight: 0.1
    temperature: 1.0

# =============================================================================
# Evaluation Protocol
# =============================================================================
evaluation:
  modes:
    - "original"  # Real images
    - "blind"     # Black images
    
  metrics:
    - accuracy  # Exact match
    - soft_accuracy  # VQA-style (matches any annotator)
    - visual_contribution  # original_acc - blind_acc
    - confidence_calibration  # If model outputs confidence
    
  cross_benchmark:
    description: "Train on one benchmark, test on all three"
    matrix:
      - train: vqav2, test: [vqav2, mmstar, mmspubench]
      - train: mmstar, test: [vqav2, mmstar, mmspubench]
      - train: mmspubench, test: [vqav2, mmstar, mmspubench]

# =============================================================================
# Expected Results & Hypotheses
# =============================================================================
hypotheses:
  H1:
    statement: "A5 > A3 on blind accuracy"
    rationale: "Confidence weighting provides better signal than uniform weighting"
    
  H2:
    statement: "A5 preserves original accuracy better than A4"
    rationale: "KL regularization prevents catastrophic forgetting"
    
  H3:
    statement: "Larger models have stronger linguistic priors (higher A0 blind acc)"
    rationale: "More parameters = more language model capacity"
    
  H4:
    statement: "Cross-benchmark generalization is high"
    rationale: "Linguistic priors are general, not benchmark-specific"
    
  H5:
    statement: "Visual contribution varies by question type"
    rationale: "Counting, spatial questions need vision; yes/no can be guessed"

# =============================================================================
# Computational Budget
# =============================================================================
compute:
  # Per ablation (2B model, 300 questions)
  training_time_hours: 1-2
  gpu_memory_gb: 24
  
  # Full study
  total_ablations: 30  # 6 ablations × 3 benchmarks + 4 scale comparisons
  estimated_total_hours: 40-60
  
  # Evaluation
  eval_time_per_model_hours: 0.5

# =============================================================================
# File Organization
# =============================================================================
directories:
  data:
    human_data: "./human_data/{benchmark}/*.csv"
    questions: "./data/{benchmark}/questions.json"
    annotations: "./data/{benchmark}/annotations.json"
    images: "./images/{benchmark}/"
    
  output:
    checkpoints: "./output/ablations/{model}/{ablation}/{benchmark}/"
    logs: "./logs/{model}/{ablation}/"
    results: "./results/{model}/{ablation}/"
    
  evaluation:
    per_ablation: "./eval_results/{model}/{ablation}/{benchmark}/"
    summary: "./eval_results/summary/"