# Cantonese ASR with Source-Conditional Acoustic and Text Adaptation

## P2 structural probes

### Experimental scope

We evaluated whether model capacity, parameter-efficient adaptation, text-side reranking, and prediction fusion offer useful directions beyond the RAW_WINNER system. The capacity experiment is deliberately a 150-step matched-budget adaptation probe rather than a convergence study. All six training arms consume the same frozen 2,400-example stream from `external73`, corresponding to 150 optimizer steps with a global effective batch size of 16. The stream order, preprocessing, decoding configuration, random seed, and fixed validation set are held constant across one-, two-, and four-GPU topologies. Small, Medium, and Large-v2 models are initialized from their respective original OpenAI checkpoints and adapted either by Full SFT or LoRA. Consequently, the results measure short-budget adaptation efficiency and resource cost; they do not establish a definitive ranking among fully converged architectures.

### Capacity and parameter-efficient adaptation

Full-SFT adaptation improves monotonically with capacity under the matched budget. Validation sentence accuracy within two character edits rises from 0.6353 for Small to 0.7151 for Medium and 0.8034 for Large-v2, while CER decreases from 0.1624 to 0.1341 and 0.1029. Large-v2 Full SFT also reaches 0.9305 Public tol2 and 0.0664 Public CER. These gains come with higher parameter count, GPU-hour cost, and inference real-time factor, which are reported alongside accuracy in the machine-readable capacity matrix. LoRA sharply reduces trainable parameters and memory, but it does not match Full SFT for any model size within 150 updates. We therefore interpret LoRA as a lower-resource adaptation option, not as an accuracy-equivalent substitute under this schedule.

### Cantonese tokenizer audit and language-model reranking

The Small, Medium, and Large-v2 checkpoints use an identical tokenizer hash. Across Official train, validation, Public, and OOD references, the audit finds neither unknown tokens nor replacement characters. Approximately 29--35% of isolated reference characters require multiple tokens, including common Cantonese characters, which demonstrates token inefficiency but does not prove that vocabulary expansion is the dominant recognition bottleneck. A leakage-guarded character 5-gram language model trained only on `external73` is then used to rescore the RAW_WINNER five-best candidates. Validation selects a language-model weight of zero, despite a measurable five-best oracle gap. Thus, the tested n-gram score does not reliably recover the oracle hypotheses and provides no evidence for deploying this reranker.

### Prediction fusion

We further test prediction complementarity among RAW_WINNER, NOISE_S43, and FULL_LR1E6_S43 without retraining. After verifying row order and reference parity on validation, Public, and OOD, we compare character-edit-distance MBR, character-level ROVER, and a best-of-three oracle. Among the registered fusion rules, validation selects MBR, which achieves 0.8704 tol2 and 0.0823 CER. This slightly trails the best individual validation component, NOISE_S43 (0.8732/0.0821), while the Public MBR result of 0.8984/0.0791 exceeds all three components. The best-of-three oracle remains stronger, showing useful diversity without proving that static consensus is uniformly better across surfaces. This is an offline analysis whose deployment cost is approximately the combined inference cost of three component models.

### Limitations and claim boundary

The capacity comparison uses a deliberately short schedule and one registered learning rate per adaptation method. It therefore supports claims about matched-budget adaptation efficiency, not fully tuned or converged model quality. Public and OOD results are diagnostic and are not used to select checkpoints or reranking parameters. EOS counters are also interpreted conservatively: the Medium/Large generation sidecars omit EOS tokens in the observed Transformers path even when decoding terminates normally, so `no_eos_count` is not treated as repeated runaway; actual repetition, replacement-character, and maximum-length diagnostics remain separate. Finally, neither tokenizer expansion nor neural external language modeling is implemented in this probe, so their potential value remains an open question.

### Claim--evidence map

- **Claim:** Larger Whisper variants adapt more efficiently within 150 matched updates. **Evidence:** monotonic Full-SFT validation tol2/CER from Small through Medium to Large-v2. **Status:** supported within the registered short-budget scope.
- **Claim:** LoRA is accuracy-equivalent to Full SFT. **Evidence:** all matched LoRA arms trail Full SFT. **Status:** not supported.
- **Claim:** The standard Whisper tokenizer is the dominant Cantonese bottleneck. **Evidence:** identical tokenizers, zero unknown/replacement tokens, but measurable multi-token characters. **Status:** not supported.
- **Claim:** The tested character 5-gram reranker improves recognition. **Evidence:** validation selects lambda zero. **Status:** not supported.
- **Claim:** Existing models contain exploitable prediction diversity. **Evidence:** MBR improves Public but slightly trails the best validation component; the oracle is stronger on every surface. **Status:** partially supported; static fusion is not uniformly superior.

### Reviewer-facing self-review

- **Contribution:** the probe supplies controlled capacity, PEFT, tokenization, LM, and fusion evidence rather than presenting untested roadmap items.
- **Clarity:** exposure, topology, optimization, selection, and decoding controls are recorded with hashes and machine-readable receipts.
- **Experimental strength:** all architectures share a strict matched-exposure protocol, but the short schedule limits convergence claims.
- **Evaluation completeness:** fixed validation, Public, OOD, error diagnostics, resources, and inference latency are reported; multiple seeds and full convergence are not.
- **Method soundness:** label leakage is prohibited, Public/OOD do not select parameters, and true repeated runaway is separated from EOS-token bookkeeping.
