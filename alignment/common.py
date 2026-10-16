"""Shared helpers for every pipeline stage: prompt format, data loading,
devices/precision, reproducibility, durable metrics/checkpoint writing."""
from __future__ import annotations

import json
import os
import random
import shutil
import time

import numpy as np
import torch

# One prompt format, used identically for SFT, both from-scratch DPO runs, and
# all Qwen generations. No chat template is used for the Qwen models.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 42

# Decoding settings shared by every generation (base, SFT, DPO, LoRA, and its base).
GENERATION_SETTINGS = {
    "do_sample": False,           # greedy; temperature / top-p unused
    "max_new_tokens": 256,
    "repetition_penalty": 1.3,    # same anti-repetition settings as the Task 2 baseline run
    "no_repeat_ngram_size": 3,
    "length_penalty": 1.0,        # inactive for greedy decoding
}

OUTPUTS = "outputs"


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_overrides(namespace: dict, stage: str) -> None:
    """Optional JSON overrides ({stage: {NAME: value}}) from $PA2_OVERRIDES.
    Unset in the real run; used only for dry runs with tiny settings."""
    path = os.environ.get("PA2_OVERRIDES")
    if not path:
        return
    with open(path) as f:
        overrides = json.load(f).get(stage, {})
    for key, value in overrides.items():
        print(f"[override] {stage}.{key} = {value}")
        namespace[key] = value


def get_devices():
    """(policy_device, reference_device). Two GPUs: policy on cuda:0, frozen
    reference on cuda:1 (manual section 10.1 memory tip)."""
    n = torch.cuda.device_count()
    if n >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    if n == 1:
        return torch.device("cuda:0"), torch.device("cuda:0")
    return torch.device("cpu"), torch.device("cpu")


def use_amp(device) -> bool:
    return torch.device(device).type == "cuda"


def autocast_ctx(device):
    """fp16 autocast on CUDA (T4 has no bf16 tensor cores); no-op on CPU."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp(device))


def load_tokenizer(name_or_path: str = BASE_MODEL, padding_side: str = "right"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def truncate_instruction(tokenizer, instruction: str, max_prompt_tokens: int) -> str:
    """Shorten the instruction (keeping its END, where the question usually is)
    so that format_prompt(instruction) is at most max_prompt_tokens tokens."""
    instruction = instruction.strip()

    def n_tokens(text):
        return len(tokenizer(format_prompt(text), add_special_tokens=False)["input_ids"])

    if n_tokens(instruction) <= max_prompt_tokens:
        return instruction
    ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    overhead = n_tokens("x") - 1
    keep = max(1, max_prompt_tokens - overhead)
    while keep > 0:
        candidate = tokenizer.decode(ids[-keep:]).strip()
        if candidate and n_tokens(candidate) <= max_prompt_tokens:
            return candidate
        keep -= 4
    raise ValueError("could not truncate instruction")


def robust_load_dataset(*args, retries: int = 4, delay: int = 20, **kwargs):
    from datasets import load_dataset
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:  # network hiccups on long runs
            if attempt == retries:
                raise
            print(f"load_dataset failed ({e}); retry {attempt}/{retries - 1} in {delay}s")
            time.sleep(delay)


def get_alpaca_eval_subset(n_prompts: int = 300, seed: int = SEED):
    """Fixed AlpacaEval subset: shuffle the 805 eval indices with
    random.Random(seed) and keep the first n_prompts."""
    rows = None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
        with open(path) as f:
            rows = [{"instruction": r["instruction"]} for r in json.load(f)]
    except Exception as e:
        print(f"alpaca_eval.json download failed ({e}); trying the parquet conversion")
    if rows is None:
        ds = robust_load_dataset(
            "parquet",
            data_files={"eval": "hf://datasets/tatsu-lab/alpaca_eval@refs%2Fconvert%2Fparquet/alpaca_eval/eval/0000.parquet"},
            split="eval",
        )
        rows = [{"instruction": r["instruction"]} for r in ds]
    if len(rows) != 805:
        raise RuntimeError(f"expected 805 AlpacaEval prompts, got {len(rows)}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [{"id": f"alpaca_{i}", "instruction": rows[i]["instruction"]} for i in indices[:n_prompts]]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MetricsLogger:
    """Appends one JSON object per line and fsyncs, so partial progress survives
    a disconnect (manual section 5.5)."""

    def __init__(self, path: str, fresh: bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        if fresh and os.path.exists(path):
            os.remove(path)

    def log(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def save_checkpoint(model, tokenizer, out_dir: str) -> None:
    """Write to a temp dir and swap in, so a crash mid-save never destroys the
    previous best checkpoint. Weights are saved in fp32 (as trained)."""
    tmp = out_dir.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)


def require_checkpoint(path: str, what: str = "checkpoint") -> None:
    if not os.path.isdir(path) or not os.listdir(path):
        raise FileNotFoundError(f"{what} not found at '{path}'. Run the earlier stage first.")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
