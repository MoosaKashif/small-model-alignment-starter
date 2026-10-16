## Judged results

| Comparison | Judge | N | Wins | Losses | Ties (of which identical) | Unknown | Failed | Win rate (excl. ties) | Win rate (ties = 0.5) | Retried pairs | A-share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HH-RLHF DPO vs SFT (HH safety eval set) | google/gemini-3.6-flash | 300 | 136 | 111 | 53 (0) | 0 | 0 | 0.551 | 0.542 | 0 | 0.447 |
| UltraFeedback DPO vs SFT (AlpacaEval-300, seed 42) | google/gemini-3.6-flash | 300 | 91 | 109 | 100 (0) | 0 | 0 | 0.455 | 0.470 | 0 | 0.420 |
| LoRA-DPO vs SmolLM2 base: persona style | google/gemini-3.6-flash | 24 | 2 | 0 | 22 (0) | 0 | 0 | 1.000 | 0.542 | 0 | 0.083 |
| LoRA-DPO vs SmolLM2 base: usefulness/safety | google/gemini-3.6-flash | 24 | 7 | 11 | 6 (0) | 0 | 0 | 0.389 | 0.417 | 0 | 0.292 |

Wins/losses are from the candidate's view (DPO model, or LoRA-DPO model). Unknown = judge returned no valid A/B/TIE for that item (kept, counted separately, excluded from win rates). Identical = both responses word-for-word the same; recorded as ties without a judge call. Failed = pair still unjudged after 3 attempts (logged in *_failures.jsonl). A-share = fraction of judge verdicts naming Response A, over judged (non-identical) pairs (position-bias check).

### HH-RLHF DPO vs SFT by category

| HH category | N | DPO wins | SFT wins | Ties | Unknown | Win rate (excl. ties) |
|---|---|---|---|---|---|---|
| benign_helpfulness | 30 | 14 | 13 | 3 | 0 | 0.519 |
| calibration_and_uncertainty | 30 | 12 | 14 | 4 | 0 | 0.462 |
| cyber_abuse | 30 | 15 | 9 | 6 | 0 | 0.625 |
| harmful_instructions | 30 | 12 | 4 | 14 | 0 | 0.750 |
| hate_or_harassment | 30 | 19 | 5 | 6 | 0 | 0.792 |
| legal_financial_safety | 30 | 11 | 12 | 7 | 0 | 0.478 |
| medical_safety | 30 | 11 | 16 | 3 | 0 | 0.407 |
| privacy_and_doxxing | 30 | 14 | 10 | 6 | 0 | 0.583 |
| safe_transformation | 30 | 14 | 12 | 4 | 0 | 0.538 |
| self_harm_support | 30 | 14 | 16 | 0 | 0 | 0.467 |
