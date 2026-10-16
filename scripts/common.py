"""Shared across every pipeline stage: prompt format + checkpoint-saving helpers.

PROMPT FORMAT (documented for the report, section 14.1/14.2):
We use a fixed Alpaca-style template, applied identically during SFT training,
both from-scratch DPO runs, and all generation/evaluation. No chat template
is used (Qwen2.5-0.5B base has no instruction-tuned chat template baked in
at the base-model stage, so a plain template is used throughout and is
documented here as the single source of truth).
"""
import os

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())

def save_checkpoint(model, tokenizer, out_dir):
    """Save in fp16 to roughly halve on-disk checkpoint size (~1.84GB -> ~920MB
    for Qwen2.5-0.5B), since disk space on Kaggle is the binding constraint once
    multiple best+final checkpoints across SFT/HH-DPO/UF-DPO accumulate.
    Training scripts cast back to float32 immediately after this call so the
    in-memory training state is unaffected."""
    os.makedirs(out_dir, exist_ok=True)
    was_training = model.training
    model.eval()
    model.half().save_pretrained(out_dir)
    model.float()
    if was_training:
        model.train()
    tokenizer.save_pretrained(out_dir)

def robust_load_dataset(*args, retries=3, delay=15, **kwargs):
    """Thin wrapper around datasets.load_dataset with retries, since a
    multi-hour full run is exposed to transient Kaggle/HF network hiccups
    that would otherwise kill the whole stage on one bad download."""
    import time
    from datasets import load_dataset
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"WARNING: load_dataset attempt {attempt}/{retries} failed ({e}); "
                  f"retrying in {delay}s..." if attempt < retries else
                  f"WARNING: load_dataset attempt {attempt}/{retries} failed ({e}); giving up.")
            if attempt < retries:
                time.sleep(delay)
    raise last_err

def require_checkpoint(path, what="checkpoint"):
    """Fail fast with a clear message instead of a cryptic HF
    'repository not found' error deep in a stack trace, if a prior stage's
    checkpoint is missing (e.g. that stage crashed or was skipped)."""
    import os as _os
    if not _os.path.isdir(path) or not _os.listdir(path):
        raise FileNotFoundError(
            f"Expected {what} at '{path}' but it does not exist or is empty. "
            f"This usually means an earlier stage did not finish successfully — "
            f"check that stage's output before re-running this one."
        )

_ALPACA_FALLBACK_PROMPTS = [
    "Explain what a prime number is in one sentence.",
    "Write a short poem about autumn.",
    "What is the capital of France?",
    "Give two tips for staying focused while studying.",
    "Summarize the plot of Cinderella in two sentences.",
    "What causes rain?",
    "Name three uses for baking soda.",
    "Describe the water cycle in two sentences.",
    "What is the difference between a crocodile and an alligator?",
    "Give a simple recipe for scrambled eggs.",
]

def get_alpaca_eval_subset(n_prompts, seed):
    """Fixed-seed AlpacaEval subset, with three layered fallbacks so the
    pipeline never hard-crashes on this: (1) load the real dataset via its
    pre-converted parquet revision, bypassing the now-unsupported loading
    script; (2) try the plain load_dataset path in case the parquet layout
    changes; (3) fall back to a small fixed prompt list so every downstream
    stage still has *something* seeded and reproducible to run on."""
    import random
    try:
        from datasets import load_dataset
        from huggingface_hub import list_repo_files
        repo_id = "tatsu-lab/alpaca_eval"
        files = list_repo_files(repo_id, repo_type="dataset", revision="refs/convert/parquet")
        eval_files = [f for f in files if "alpaca_eval" in f and f.endswith(".parquet")]
        if not eval_files:
            raise RuntimeError("no alpaca_eval parquet files found in refs/convert/parquet")
        ds = load_dataset(
            "parquet",
            data_files={"eval": f"hf://datasets/{repo_id}@refs%2Fconvert%2Fparquet/{eval_files[0]}"},
            split="eval",
        )
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        selected = indices[:n_prompts]
        prompts = [{"id": f"alpaca_{i}", "instruction": ds[i]["instruction"]} for i in selected]
        print(f"Loaded {len(prompts)} prompts from tatsu-lab/alpaca_eval via parquet revision (seed={seed})")
        return prompts
    except Exception as e1:
        print(f"WARNING: parquet-revision load failed ({e1}); trying plain load_dataset...")
        try:
            from datasets import load_dataset
            ds = load_dataset(repo_id, "alpaca_eval", split="eval")
            indices = list(range(len(ds)))
            random.Random(seed).shuffle(indices)
            selected = indices[:n_prompts]
            prompts = [{"id": f"alpaca_{i}", "instruction": ds[i]["instruction"]} for i in selected]
            print(f"Loaded {len(prompts)} prompts from tatsu-lab/alpaca_eval via plain load_dataset (seed={seed})")
            return prompts
        except Exception as e2:
            print(f"WARNING: plain load_dataset also failed ({e2}); using fixed fallback prompt list. "
                  f"NOTE: document this substitution in your report if it persists into the full run.")
            random.Random(seed).shuffle(_ALPACA_FALLBACK_PROMPTS)
            n = min(n_prompts, len(_ALPACA_FALLBACK_PROMPTS))
            return [{"id": f"fallback_{i}", "instruction": _ALPACA_FALLBACK_PROMPTS[i]} for i in range(n)]

def get_prompt_text(record):
    """Extract instruction/prompt text from a record whose key name may vary
    across provided eval-set files (prompt / instruction / input / text)."""
    for key in ("instruction", "prompt", "input", "text", "question"):
        if key in record and record[key]:
            return record[key]
    raise KeyError(f"No known prompt field found in record keys: {list(record.keys())}")
