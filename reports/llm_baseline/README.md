# LLM baseline (ALLaM-7B-Instruct-preview, zero-shot, run locally)

Script: `scripts/llm_baseline_allam.py` (prompts, decoding, truncation and scoring are all in the file).

| file | content |
|---|---|
| `results_allam.json` | TEST metrics for S1 prompt v1, S1 prompt v2 and S2, with document-clustered bootstrap CIs and paired contrasts against B0/B1/B2 and the encoder seeds |
| `s1_test_v1_parsed_predictions.json` | per region: id, parsed role codes, truncation flag (Prompt A in the paper: two-slot format example) |
| `s1_test_v2_parsed_predictions.json` | same, Prompt B (no format example) |
| `s2_test_parsed_predictions.json` | per sentence: id, parsed role code, truncation flag |

Only parsed role codes are published, not the model's raw text output, and no gold labels or judgment text.
An empty `pred` list is an answer without a valid role code; it is scored as an empty prediction in S1 and
falls back to the majority TRAIN role in S2 (reported in `results_allam.json`).
