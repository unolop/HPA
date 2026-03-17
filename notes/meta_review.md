# Meta Review Notes

This study investigates answer patterns of humans, LMs, and VLMs to vision-language multimodal questions, but without visual information. This study creates a dataset of human responses under these conditions, which is then compared with LLMs' and VLMs' responses. The experimental results show that base LLMs and VLMs are generally misaligned with humans and also show that this misalignment can be mitigated through fine-tuning.

Analyzing or simulating error patterns itself has been a central topic of psycholinguistics or cognitive modeling; for example, it is a well-known research topic how humans arrive at erroneous answers through superficial, fast thinking, and what mechanisms or rules govern it. Therefore, studying the errors or speculation patterns of humans themselves is sound, and the resource will be useful even from interdisciplinary perspectives. Language models may also be integrated into such studies to test, for example, whether humans' processing biases reflect corpus frequency (if one sees LMs as a model of corpus frequency [1]).

Furthermore, this study can also be connected to the diagnosis of possible shortcuts inherent in vision-language benchmarks. Just as the surprising performance of the hypotheses-only baseline has been discussed in NLI [2], it is crucial first to explore "how much an image is actually needed" in the benchmark. If the text-only baseline could somehow solve the task without visual information, then the next question will be whether this is a natural consequence, and human speculation patterns and performance will be a good reference point.
Connecting this study to these topics may make this paper more convincing, rather than merely associating it with hallucination or alignment in the context of LLMs. 

[1] McCoy, R. Thomas, et al. "Embers of autoregression show how large language models are shaped by the problem they are trained to solve." Proceedings of the National Academy of Sciences 121.41 (2024): e2322420121.
[2] Poliak, Adam, et al. "Hypothesis only baselines in natural language inference." Proceedings of the seventh joint conference on lexical and computational semantics. 2018.  

Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.

---

## Analysis of Reviewer Suggestions

### Core Reframing Request
Move away from "hallucination / LLM alignment" framing.
Move toward **two stronger framings**:
1. **Benchmark diagnosis** — analogous to hypothesis-only NLI baselines [2]
2. **Corpus frequency exploitation** — LMs as models of training distribution [1]

### Reference [1] — Embers of Autoregression (McCoy et al., PNAS 2024)
- LLMs are shaped by next-word prediction objective over internet text
- Three factors predict success/failure: probability of (task, output, input)
- Models bias toward high-probability outputs regardless of task requirements
- **Key implication for our work:** blind VQA accuracy = measuring corpus frequency exploitation
  - When VLMs answer correctly without images, they are completing high-probability sequences
  - This is the mechanistic explanation for our blind accuracy results
- **Experiment to add:** correlate blind accuracy with answer frequency in VQA training distribution
  - High correlation = direct evidence of corpus-frequency mechanism
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11474099/

### Reference [2] — Hypothesis-Only Baselines (Poliak et al., *SEM 2018)
- In NLI: model trained only on hypothesis (ignoring premise) beats majority baseline across 10 datasets
- Revealed annotation artifacts / statistical biases in NLI benchmarks
- **Direct analogy to our work:** image = premise, question = hypothesis; blind VQA = hypothesis-only NLI
- **Language to adopt in paper:**
  > "Analogous to hypothesis-only baselines in NLI [Poliak 2018], we evaluate VLMs using
  > question-only inputs to diagnose the extent to which VQA benchmarks can be solved
  > without visual grounding."
- Link: https://aclanthology.org/S18-2023/

### What Makes Our Paper Stronger Than Both References
- We have **human blind VQA responses** as a natural reference point
- Humans do NOT have corpus-frequency bias in the same way → human vs. model divergence directly measures the training-distribution effect
- Neither McCoy nor Poliak have a human comparison — this is our unique contribution

### Response Plan
| Concern | Action |
|---------|--------|
| Reframe away from hallucination | Add benchmark diagnosis framing in intro; cite Poliak analogy explicitly |
| Connect to corpus frequency | Add experiment correlating blind acc with answer frequency; cite McCoy |
| Psycholinguistics angle | Add paragraph in related work on human fast/slow thinking and error patterns |
| Human data as reference point | Strengthen human vs. model comparison section; emphasize this is unique contribution |
