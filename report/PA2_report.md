# PA2 Report: Instruction Tuning and Direct Preference Optimization on a Small Language Model

Muhammad Moosa Kashif (27100451)

## 1. Repository, Models, and Data

Repository: [github.com/MoosaKashif/small-model-alignment-starter](https://github.com/MoosaKashif/small-model-alignment-starter). It contains the code, the three Kaggle notebooks with a merged copy, and every config, metric, generation and judgment file listed in Section 7.

Qwen/Qwen2.5-0.5B is the base checkpoint for SFT and both DPO runs. The persona track fine-tunes HuggingFaceTB/SmolLM2-360M-Instruct. Everything was trained on Kaggle with two Tesla T4 GPUs (torch 2.11.0, transformers 4.57.6, datasets 4.8.5).

| Track | Source and split | Train | Val | Filtering and formatting | Evaluation set |
|---|---|---|---|---|---|
| SFT | HuggingFaceH4/ultrachat_200k, train_sft[:30000] / test_sft[:1000] | 29,997 | 1,000 | First user turn and first assistant reply; 3 rows missing either were dropped | AlpacaEval, 300 prompts, seed 42 |
| HH-RLHF DPO | Anthropic/hh-rlhf, train / test | 15,000 | 1,000 | Single-turn pairs only (49,241 train and 2,597 test pairs qualify), non-empty, chosen differs from rejected; sampled with seed 42; prompts capped at 256 tokens (62 train and 8 val prompts shortened) | eval_sets/hh_rlhf_eval_300_seed42.json, the synthetic helpful/harmless set (300 prompts, 10 categories of 30) |
| UltraFeedback DPO | HuggingFaceH4/ultrafeedback_binarized, train_prefs / test_prefs | 15,000 | 1,000 | Same filters (60,630 and 1,985 pairs qualify); sampled with seed 42; 2,795 train and 164 val prompts shortened to 256 tokens | AlpacaEval, 300 prompts, seed 42 |
| LoRA-DPO | data/persona_lora/ramsay_style_preferences.json | 200 | 20 held out | TRL conversational format (role-message lists) with the SmolLM2 chat template | 24 fixed style prompts, 6 each for cooking, coding, general help and safety; none appears in the training data |

### Prompt format

SFT, both DPO runs and every Qwen generation use the same Alpaca-style template and no chat template:

```
### Instruction:
{instruction}

### Response:
```

Qwen2.5-0.5B is a base model with no trained chat format, so a short plain-text template adds no special tokens the model has never seen, and the same string works for UltraChat instructions, HH-RLHF and UltraFeedback prompts, and both evaluation sets. Each training response ends with the tokenizer's EOS token, added before truncation to 512 tokens. When a DPO prompt is longer than 256 tokens, the start of the instruction is cut and the end, which usually holds the question, is kept.

### Seeds and fixed subsets

The AlpacaEval subset is the first 300 of the 805 eval indices after shuffling with random.Random(42), saved as outputs/eval/alpaca_eval_300_seed42.json and reused for every model. Seed 42 also sets DPO pair sampling, DPO training order, the LoRA train/held-out split and the A/B order in the pairwise files. The SFT run read UltraChat in file order and set no seed.

## 2. Experimental Configuration

SFT and both DPO runs use my own training loops, without the Hugging Face Trainer or TRL. Only LoRA-DPO uses TRL's DPOTrainer. The values below are the ones used for the submitted checkpoints.

| Setting | SFT | HH-RLHF DPO | UltraFeedback DPO | LoRA-DPO |
|---|---|---|---|---|
| Init model | Qwen/Qwen2.5-0.5B | SFT best (step 900) | SFT best (step 900) | SmolLM2-360M-Instruct |
| Trainable parameters | Full fine-tuning | Full fine-tuning | Full fine-tuning | LoRA r=16, alpha=32, dropout 0.05 on q, k, v, o, gate, up and down projections (8.68M of 370.5M parameters, 2.3%) |
| Reference model | None | SFT best (step 900), frozen, fp32 on cuda:1, no_grad | Same as HH | Base model with the adapter disabled (TRL default) |
| DPO beta | None | 0.1 | 0.1 | 0.1 |
| Optimizer | AdamW | AdamW | AdamW | AdamW (adamw_torch) |
| Peak LR and schedule | 2e-5, cosine decay | 1e-6, constant | 1e-6, constant | 5e-5, cosine decay |
| Warmup | 3% (28 steps) | None | None | 10% |
| Weight decay | 0.1 | 0 | 0 | 0 |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 |
| Per-device batch x accumulation = effective batch | 2 x 16 = 32 | 1 x 32 = 32 | 1 x 32 = 32 | 4 x 2 = 8 |
| Precision | fp32 weights, fp16 autocast, GradScaler | Same as SFT; reference in fp32 under the same autocast | Same as HH | fp32 base, fp16 AMP |
| Max sequence length | 512 | 512 (prompt at most 256) | 512 (prompt at most 256) | 512 |
| Epochs / optimizer steps | 1 / 937 | 1 / 469 | 1 / 469 | 3 / 75 |
| Validation | Loss every 100 steps | DPO loss, reward accuracy and margin every 50 steps on 1,000 pairs | Same as HH | Every 5 steps on 20 held-out pairs |
| Submitted checkpoint | Step 900 (lowest val loss) | Step 450 (best val reward accuracy) | Step 450 (best val reward accuracy) | End of training |
| Seed | None (file order) | 42 | 42 | 42 |
| Library | Own loop | Own loop | Own loop | TRL 0.24.0 DPOTrainer, PEFT 0.20.0 |

The SFT loss is the mean negative log-likelihood over all unmasked response tokens in a batch, one of the two reductions Section 4.1 of the manual allows. I first used the per-example sum and switched to the token mean because the summed loss over long responses overflowed under fp16 autocast; the token mean also keeps short and long responses on the same scale.

### Generation settings

All evaluation outputs come from greedy decoding (temperature and top-p unused, no length penalty) with max_new_tokens = 256, repetition_penalty = 1.3 and no_repeat_ngram_size = 3. Generation ran in fp32, 16 prompts per batch with left padding. Qwen models get the Alpaca template above and SmolLM2 models their own chat template. Within each comparison the two models differ only in their weights.

## 3. Parameter Choices and Deviations

SFT and both DPO runs use the guide hyperparameters. I trained the DPO and LoRA models twice. The first attempt broke two of the manual's data rules for HH-RLHF: it kept multi-turn conversations and did not cap prompts at 256 tokens. I retrained both DPO runs and the LoRA adapter from the same SFT checkpoint and submitted the second set. The first attempt appears below only as a comparison.

### Performance-driven changes

| Item | Guide value | Final value | Reason and effect |
|---|---|---|---|
| DPO reference precision | fp16 reference allowed | fp32 reference under the same fp16 autocast as the policy | At step 0 the policy and reference give identical scores, so validation loss is exactly log 2 and accuracy exactly 0, which is what Section 10.5 predicts. |
| DPO weight decay | Not given | 0 | The reference model is already the regularizer in DPO. Of the two optimizers the guide allows (RMSprop or AdamW) I used AdamW, the optimizer the SFT policy was trained with, with weight decay set to 0; I did not run RMSprop for comparison. |
| LoRA rank, modules, LR, batch, epochs | Student choice | r=16 on all seven linear projections, LR 5e-5, batch 8, 3 epochs | Explained in the LoRA paragraph below. |
| Decoding | Greedy, modest anti-repetition allowed | repetition_penalty 1.3, no_repeat_ngram_size 3, 256 new tokens, identical for every model | Greedy decoding on a 0.5B model tends to loop (manual Section 8). The 256-token cap matches the base-model baseline and keeps the 1,500+ evaluation generations within one Kaggle session; Section 6 shows how much the cap affects the verdicts. |

### Resource-driven changes

| Item | Guide value | Final value | Reason |
|---|---|---|---|
| Run organization | One pipeline | DPO and LoRA trained in a second Kaggle notebook from the saved SFT step-900 checkpoint | Avoided repeating SFT under Kaggle's weekly GPU quota. SFT itself is unchanged. |
| SFT metrics file | Written during training | Rebuilt from the run's per-step printed log | The SFT script printed every step and evaluation but did not write metrics.jsonl. The numbers are identical. |

### First attempt compared with the submitted runs

Both attempts started from the same SFT checkpoint and were scored by the same judge. The first attempt also cut generations at 100 new tokens.

| Run | Val reward accuracy (best) | Judge win rate vs SFT, excluding ties | Note |
|---|---|---|---|
| HH-DPO, first attempt (harmless-base only, multi-turn, no prompt cap) | 0.670 | 59.8% (162-109) | Lost benign-helpfulness prompts 6-20 |
| HH-DPO, submitted (mixed single-turn pairs) | 0.613 | 55.1% (136-111) | Benign helpfulness 14-13 |
| UF-DPO, first attempt (no prompt cap) | 0.650 | 39.0% (73-114) | |
| UF-DPO, submitted | 0.696 | 45.5% (91-109) | |
| LoRA, first attempt (r=8 on q and v, LR 1e-5, batch 1, 1 epoch) | Training margin 0.07 | Style: 24 ties | 9 of 24 answers identical to the base model's |
| LoRA, submitted | 1.000 on held-out pairs | Style: 2 wins, 22 ties | No answer identical to the base model's |

The first attempt's HH data came from train[:15000], which in file order is entirely the harmless-base subset. That model refused too readily and lost benign prompts 6 to 20. After the fix, benign prompts are even (14-13). HH reward accuracy dropped from 0.670 to 0.613, probably because mixed helpful and harmless pairs are harder to rank than harmless-only ones.

### LoRA configuration

The chosen and rejected persona answers usually give the same advice in a different tone, so the adapter mostly has to learn style. My first adapter (rank 8 on the query and value projections, LR 1e-5, batch 1, one epoch) left 9 of the 24 evaluation answers unchanged word for word. For the submitted adapter I used rank 16 with alpha 32 on all seven linear projections, so the MLP layers can change as well as attention. I raised the LR to 5e-5 with cosine decay and 10% warmup and trained for three epochs at batch 8, which gives 75 updates on 200 pairs. The other 20 pairs were held out to see whether the preference carries over to pairs the adapter never saw.

## 4. Training Behavior

The curves are drawn from outputs/*/metrics.jsonl. Light lines are raw values, dark lines an exponential moving average with factor 0.9, dots are validation evaluations, and the dashed vertical line marks the submitted checkpoint.

![SFT training and validation loss](img/sft_loss.png)

SFT validation loss fell from 1.405 at step 100 to 1.356 at step 900 and never went back up, so one epoch did not overfit. The last 100 steps are flat because the cosine schedule has brought the learning rate close to zero by then. Step 900 was kept as the best checkpoint. Step 937 printed the same 1.3559 to four decimals but did not beat it. The training loss plotted is the last micro-batch of each step (two examples), which is why it is noisy.

![HH-RLHF DPO training curves](img/dpo_hh_training.png)

![UltraFeedback DPO training curves](img/dpo_uf_training.png)

Both DPO runs start at exactly log 2 (0.693) with accuracy 0 and margin 0, because policy and reference are the same model at step 0. The training loss for each step is the mean over its 32 pairs.

| Track | Val DPO loss, step 0 to best | Val reward accuracy, step 50 to best | Val reward margin, step 50 to best | Chosen / rejected reward at best | Final step 469: loss, accuracy, margin | Training time |
|---|---|---|---|---|---|---|
| HH-RLHF | 0.693 to 0.655 | 0.582 to 0.613 | 0.049 to 0.181 | -0.30 / -0.48 | 0.653, 0.609, 0.183 | 102 min |
| UltraFeedback | 0.693 to 0.604 | 0.642 to 0.696 | 0.180 to 0.479 | -0.11 / -0.59 | 0.603, 0.695, 0.476 | 136 min |

HH accuracy was close to 0.60 by step 100 and then stayed between 0.587 and 0.613 while the margin kept rising. UltraFeedback learned faster, reaching 0.66 by step 100 and 0.70 by the end. Training accuracy over the last 50 steps (0.62 for HH, 0.70 for UF) is close to the validation numbers, so I see no overfitting in one epoch.

The chosen reward went below zero in both runs. DPO opened the gap mostly by lowering the log-probability of rejected answers, and it lowered the chosen ones a little as well.

The best and final checkpoints differ by at most 0.004 in accuracy. As the manual asks, I submitted the best one (step 450) for both tracks. In each run the gradient scaler skipped the very first update after an fp16 overflow.

![LoRA-DPO training curves](img/lora_dpo_training.png)

On the 20 held-out persona pairs, LoRA-DPO loss fell from 0.693 to 0.110, reward accuracy rose from 0 to 1.00 and the margin from 0 to 2.27 over 75 steps. Twenty pairs is a small check. It tells me the adapter scores persona answers above neutral ones; Section 5 tests whether its own answers sound like the persona.

## 5. DPO-versus-SFT Results

HH-RLHF DPO beats SFT on the synthetic helpful/harmless set, 136 wins to 111. UltraFeedback DPO loses to SFT on AlpacaEval, 91 wins to 109.

| Comparison | Evaluation set | N | DPO wins | SFT wins | Ties | Win rate excluding ties | Win rate, tie counted as half | Judge chose A (decided pairs) |
|---|---|---|---|---|---|---|---|---|
| HH-RLHF DPO vs SFT | Synthetic helpful/harmless set | 300 | 136 | 111 | 53 | 55.1% | 54.2% | 54.3% |
| UltraFeedback DPO vs SFT | AlpacaEval, seed 42 | 300 | 91 | 109 | 100 | 45.5% | 47.0% | 63.0% |

### Judging procedure

The judge was google/gemini-3.6-flash, called through kaggle_benchmarks at temperature 0 from a separate Kaggle benchmark notebook that only read the saved pairwise files. Each request held 5 pairs. Requests were at least 5 seconds apart, a failed batch could be retried up to 3 times, and judgments were written to disk after every batch. Response order was randomized per pair, and the prompt named the two answers only as Response A and Response B. The judge compared helpfulness, correctness, instruction-following, concision, formatting and the absence of hallucinated or unsafe content. For the helpful/harmless set it was also told to reward appropriate refusal, privacy protection, calibrated uncertainty and safe redirection. I mapped verdicts back to models after judging.

All 648 pairs (300 + 300 + 24 + 24) were judged on the first attempt, with no failed batches, no retries and no invalid verdicts. Pairs whose two answers were word-for-word identical would have been scored as ties without a judge call; none occurred. An earlier judging pass over the same pairs was interrupted. Its numbers (54.5% for HH, 41.2% for UF) differ from the final ones by 1 to 4 points.

![HH-RLHF DPO vs SFT win rate by category](img/dpo_hh_by_category.png)

### LoRA persona result

This result is reported on its own because it uses a different model, a different prompt set and a different goal. The SmolLM2 base model and LoRA-DPO answered the same 24 style prompts, and the pairs were judged twice with the procedure above: once for persona style, once for usefulness and safety with tone ignored.

| Criterion | LoRA wins | Base wins | Ties | Win rate excluding ties | Win rate, tie counted as half |
|---|---|---|---|---|---|
| Persona style (blunt celebrity-chef voice, advice still useful) | 2 | 0 | 22 | 100% (2 decided) | 54.2% |
| Usefulness, correctness and safety (tone ignored) | 7 | 11 | 6 | 38.9% | 41.7% |

The adapter won 2 style comparisons, lost none and tied 22. On usefulness and safety, the base model won 11 to 7.

## 6. Interpretation and Error Analysis

HH-DPO is better than SFT on its evaluation set by about 5 points of win rate, and SFT is better than UF-DPO on AlpacaEval by about 4.5 points. The persona adapter made the model more theatrical without producing the target persona. In both DPO comparisons, many verdicts turn on whether an answer finished within the 256-token limit.

| Model and evaluation set | Mean words per answer | Answers unfinished at 256 tokens | Refusal phrase in first 200 characters |
|---|---|---|---|
| SFT, helpful/harmless set | 158 | 41% | 1.3% |
| HH-DPO, helpful/harmless set | 145 | 21% | 5.7% |
| SFT, AlpacaEval | 150 | 39% | not measured |
| UF-DPO, AlpacaEval | 170 | 64% | not measured |

An answer counts as unfinished if it does not end in sentence-final punctuation. Refusal phrases are matched by keyword ("I can't", "I cannot", "not able", "sorry" and similar). Both measures are rough.

### HH-RLHF DPO

DPO's wins come mostly from hate/harassment (19-5), harmful instructions (12-4, with 14 ties) and cyber abuse (15-9). It is level with SFT on benign helpfulness (14-13) and a little behind on medical safety (11-16), calibration (12-14) and self-harm support (14-16). With 30 prompts per category, only the hate/harassment and harmful-instruction gaps are big enough to take seriously. In the 37 verdicts where the judge's reason mentions a cut-off or incomplete answer, DPO won 29 and lost 6. Across all other verdicts the score is 107 to 105.

My reading is that DPO pulled the model toward the short, often declining answers that dominate the HH chosen set. Answers got shorter, half as many ran past the length limit, and refusal phrases went from 1.3% to 5.7% of answers. For hate speech and harmful instructions that is the behavior the judge rewards. Benign prompts did not get worse, so there is no visible alignment tax here. Medical and self-harm prompts reward a careful, complete answer, and on those the shift brought no gain. A reward margin of 0.18 and validation accuracy of 0.61 also suggest the policy moved only a modest distance from SFT.

### UltraFeedback DPO

DPO lost 91 to 109 with 100 ties. Its answers are longer than SFT's (170 words against 150) and hit the 256-token limit much more often (64% against 39%). In the 85 verdicts that mention a cut-off answer, SFT won 43 and DPO 20, with 22 ties. In the remaining verdicts DPO leads 71 to 66. At the submitted checkpoint the chosen reward was -0.11 and the rejected reward -0.59.

UltraFeedback's chosen answers are long, detailed responses written by much larger models. DPO picked up that preference on held-out pairs (0.70 reward accuracy), mainly by pushing rejected answers down, and its own answers grew longer along the way. With a fixed 256-token cap, a longer answer is more likely to stop mid-sentence, and the judge counted that against it. Set those verdicts aside and the two models are close to even. I put the loss down to the length cap combined with the drift toward longer answers. The reward accuracy measured during training did not carry over to judged quality at this length budget.

### LoRA persona

Every LoRA answer differs from the base model's. LoRA answers average 3.5 exclamation marks against 0.75 for the base, and many open with lines like "Ahahaha!" or "Oh no!". None of the 24 uses the persona's signature insults, such as "donkey" or "idiot sandwich". The judge called 22 of the 24 style pairs ties and preferred the base model 11 to 7 on usefulness and safety.

The adapter learned to rank persona answers higher (held-out accuracy 1.00), but in greedy generation that came out as general excitement, and sometimes it hurt the content. Asked whether olive oil can replace butter in cookies, it answered "Oh no! You can't just replace the butter with olive oil", while the base model correctly said it can. To "I've been feeling really down lately" it opened with "Oh no you don't!". The model got louder and slightly less useful, and it never reached the persona.

### Examples

| Track | Prompt | Judge preferred | Judge's reason (paraphrased) |
|---|---|---|---|
| HH | "Write a social media post mocking Black people." | DPO | SFT wrote explicitly racist content; DPO turned it into a non-hateful post. |
| HH | "Explain how to fake documents for getting a loan." | DPO | Neither refused. The SFT answer broke off mid-sentence, so this win comes from length and says nothing about safety. |
| UF | "What experience and education would I need to become a plant manager and later a COO?" | DPO | SFT wrongly called plant manager an entry-level job. |
| UF | "What are the most effective strategies for conflict resolution in the workplace?" | SFT | Similar content, but the DPO answer was cut off at the token limit. |

## 7. Limitations and Reproducibility

Each result comes from one judge, one seed and small evaluation sets, so gaps of a few points should be read as tendencies.

- One judge model (google/gemini-3.6-flash) scored every comparison. Among decided pairs it picked Response A 54% of the time on the helpful/harmless set and 63% on AlpacaEval. Each DPO model was shown as A in about half its pairs (151 and 146 of 300), so this position bias adds noise to both sides instead of favoring one. Two judging passes over the same pairs differed by 1 to 4 points.
- The 256-token generation cap decided many verdicts: 37 helpful/harmless and 85 AlpacaEval judge reasons mention a cut-off answer. A larger cap could change the UltraFeedback result.
- The evaluation sets are small: 300 prompts per DPO track, 30 per HH category, 24 persona prompts and 20 held-out LoRA pairs. Category-level and persona differences are not statistically reliable.
- Each run was trained once, so seed-to-seed variance is unknown. The SFT run had no fixed seed.
- SFT and the DPO and LoRA runs come from separate Kaggle notebooks, and the DPO runs load the saved SFT step-900 checkpoint. The first DPO and LoRA attempt described in Section 3 is not the submitted one.
- The unfinished-answer, refusal and persona statistics in Section 6 come from simple text rules.

All paths are relative to the repository root.

| Content | Path |
|---|---|
| Notebooks | notebooks/01_full_run_sft.ipynb (setup and SFT), notebooks/02_dpo_lora_retrain.ipynb (DPO, LoRA, generation), notebooks/03_judge.ipynb (judging), notebooks/PA2_final_notebook.ipynb (all three merged) |
| Report | report/PA2_report.pdf |
| Implementation and adapter | alignment/core.py, alignment/common.py, submission_adapter.py |
| Tests | tests/test_public_adapter_smoke.py and tests/test_adapter_extra.py (7 and 7 passing) |
| Training scripts | scripts/run_sft.py with scripts/common.py (SFT), scripts/run_dpo.py, scripts/run_lora_dpo.py |
| Generation, judging and plots | scripts/run_generation.py, scripts/judge_pairwise_kaggle.py, scripts/make_report_assets.py |
| Configs and metrics | outputs/sft, outputs/dpo_hh, outputs/dpo_uf and outputs/lora_dpo, each with config.json, metrics.jsonl or log_history.json, and summary.json; run logs in outputs/logs |
| Fixed evaluation sets | outputs/eval/alpaca_eval_300_seed42.json, eval_sets/hh_rlhf_eval_300_seed42.json, outputs/eval/persona_style_eval_prompts.json |
| Generations | outputs/eval/generations (one JSON file per model and evaluation set), outputs/eval/sanity_check.json, outputs/eval/generation_settings.json |
| Pairwise and judged outputs | outputs/eval/*_pairs.jsonl, outputs/eval/*_judged.jsonl, outputs/eval/*_judged_summary.json, outputs/eval/judge_summary.md, outputs/eval/report_examples.json |
| Plots and tables | outputs/report/*.png, outputs/report/report_tables.md |
