# RAW_WINNER P2 Structural Probe

> Claim boundary: this is a 150-step matched-budget adaptation probe, not a fully converged architecture ranking.

## Capacity and PEFT results

| Arm | Method | Params (M) | Trainable | Val tol2 | Val CER | Public tol2 | Public CER | OOD tol2 | OOD CER | GPU-h | Peak alloc (GB) | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SMALL_FULL | full_sft | 241.7 | 99.523% | 0.6353 | 0.1624 | 0.7142 | 0.1530 | 0.2810 | 0.3820 | 0.075 | 4.99 | 0.0268 |
| SMALL_LORA | lora | 242.6 | 0.365% | 0.5385 | 0.2065 | 0.6363 | 0.1848 | 0.2565 | 0.4122 | 0.091 | 2.10 | 0.0317 |
| MEDIUM_FULL | full_sft | 763.9 | 99.799% | 0.7151 | 0.1341 | 0.8537 | 0.0956 | 0.2875 | 0.3842 | 0.155 | 7.12 | 0.0403 |
| MEDIUM_LORA | lora | 766.2 | 0.308% | 0.5043 | 0.2326 | 0.6295 | 0.1920 | 0.2380 | 0.4550 | 0.297 | 3.93 | 0.0537 |
| LARGE_V2_FULL | full_sft | 1543.3 | 99.876% | 0.8034 | 0.1029 | 0.9305 | 0.0664 | 0.3325 | 0.3459 | 1.060 | 10.80 | 0.0607 |
| LARGE_V2_LORA | lora | 1547.2 | 0.254% | 0.7892 | 0.1129 | 0.8921 | 0.0864 | 0.3125 | 0.3623 | 0.566 | 6.58 | 0.0720 |

## Findings

- Capacity improves matched-budget adaptation efficiency monotonically: Small Full `0.6353/0.1624`, Medium Full `0.7151/0.1341`, Large-v2 Full `0.8034/0.1029` (tol2/CER).
- Large-v2 Full passes the registered capacity threshold versus Small Full: `True`.
- LoRA materially reduces trainable parameters and memory, but under this short budget it does not match Full SFT accuracy; this is adaptation-efficiency evidence, not a convergence claim.
- All three architectures use the same tokenizer hash (`cd04663643cab3e7a63a1f4cf2d7913e957183dbfb1b3aaebd97992083b8af36`); there are no UNKs or replacement characters in the audited references, but roughly 29–35% of individual reference characters tokenize to multiple tokens, so token inefficiency is measurable without proving it is the dominant error source.
- Leakage-guarded character 5-gram reranking selects `lambda=0.0`; therefore the LM did not improve the validation selection over ASR scores.
- Among the two registered static fusion rules, validation selects `mbr`. MBR reaches validation tol2 `0.8704` and CER `0.0823`, slightly trailing the best component NOISE_S43; on Public it reaches `0.8984/0.0791` and exceeds all three components.
- Large-v2 Full demonstrates strong short-budget capacity and Public adaptation, but it is not a replacement for RAW_WINNER: its fixed-validation tol2 remains materially lower and its OOD CER is worse. It requires a domain-aligned continuation study before deployment.
- `no_eos_count` is reported separately from true repeated runaway. Medium/Large generation sidecars omit EOS in this Transformers path even when decoding terminates normally; repeated-runaway, max-length and replacement-character diagnostics remain the stability indicators.

## Registered questions

1. **Does model capacity improve 150-step adaptation efficiency?** Yes. Full-SFT validation improves monotonically from Small to Medium to Large-v2, while inference RTF and parameter cost increase monotonically as well.
2. **What is the Full-SFT/LoRA trade-off?** LoRA reduces trainable parameters and peak memory substantially, but every LoRA arm is worse than its matched Full-SFT arm under this short budget. This does not establish the fully converged ordering.
3. **Is the standard Whisper tokenizer a demonstrated Cantonese bottleneck?** No. The tokenizers and hashes are identical and audited references contain no unknown or replacement tokens. Multi-token encoding of Cantonese characters is measurable, but this probe does not show that vocabulary modification would improve ASR.
4. **Does the character LM exploit the 5-best oracle space?** No. Validation selects lambda zero even though an oracle gap exists, so this training-text-only 5-gram score does not identify the better hypotheses reliably.
5. **Are the three existing models complementary?** Partially. The oracle and the Public MBR gain demonstrate diversity, but validation MBR slightly trails the best component. Static consensus is therefore not consistently superior across surfaces.

## Limitations

- The 150-step matched-exposure protocol measures adaptation efficiency, not convergence, and must not be cited as a definitive Small/Medium/Large architecture ranking.
- Full and LoRA learning rates are method-specific registered values; the experiment compares practical short-budget recipes rather than exhaustively optimized hyperparameters.
- Public and OOD are diagnostic surfaces. Only fixed validation is used for endpoint and reranker selection.
- The fusion result is an offline multi-model analysis and carries approximately the combined inference cost of its three component systems.

## Reproducibility

- Fixed exposure: 2,400 unique `external73` rows, seed 42, globally identical 16-example optimizer steps across 1/2/4 GPU topologies.
- Decode: beam 2, no-repeat n-gram 4, repetition penalty 1.05, max_length 225, Chinese transcription.
- Full configuration, checkpoint weight surfaces, SHA-256 values, per-surface errors and resource receipts are in `artifacts/raw_winner_p2/final/capacity_matrix.json`.
