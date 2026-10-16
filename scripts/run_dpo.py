"""Task 4: from-scratch DPO (no TRL / HF Trainer). The same implementation is
used for both tracks:
    python scripts/run_dpo.py --track hh   # Anthropic/hh-rlhf
    python scripts/run_dpo.py --track uf   # HuggingFaceH4/ultrafeedback_binarized
Policy = SFT best checkpoint (trainable, fp32 weights, fp16 autocast + GradScaler).
Reference = a second, frozen copy of the SFT best checkpoint (no_grad)."""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM

from alignment import core
from alignment.common import (OUTPUTS, SEED, MetricsLogger, apply_overrides, autocast_ctx, format_prompt,
                              get_devices, load_tokenizer, require_checkpoint, robust_load_dataset,
                              save_checkpoint, set_seed, truncate_instruction, use_amp, write_json)

# ---------------- configuration ----------------
SFT_BEST_CHECKPOINT = os.path.join(OUTPUTS, "sft", "best")   # SFT step-900 checkpoint from the full run
N_TRAIN = 15_000
N_VAL = 1_000
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 32
LR = 1e-6
BETA = 0.1
MAX_GRAD_NORM = 1.0
EPOCHS = 1
EVAL_EVERY = 50
EVAL_BATCH_SIZE = 2
# ------------------------------------------------
apply_overrides(globals(), "dpo")

HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def split_hh(text):
    """Single-turn HH transcript -> (instruction, response); None if multi-turn."""
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    h, a = text.index(HUMAN), text.index(ASSISTANT)
    if h > a:
        return None
    return text[h + len(HUMAN):a].strip(), text[a + len(ASSISTANT):].strip()


def hh_rows(split):
    ds = robust_load_dataset("Anthropic/hh-rlhf", split=split)
    rows = []
    for ex in ds:
        c, r = split_hh(ex["chosen"]), split_hh(ex["rejected"])
        if c is None or r is None or c[0] != r[0]:
            continue
        if c[0] and c[1] and r[1] and c[1] != r[1]:
            rows.append((c[0], c[1], r[1]))
    return rows


def uf_rows(split):
    ds = robust_load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    rows = []
    for ex in ds:
        ch, rj = ex["chosen"], ex["rejected"]
        if len(ch) != 2 or len(rj) != 2 or ch[1]["role"] != "assistant" or rj[1]["role"] != "assistant":
            continue
        prompt, c, r = ex["prompt"].strip(), ch[1]["content"].strip(), rj[1]["content"].strip()
        if prompt and c and r and c != r:
            rows.append((prompt, c, r))
    return rows


TRACKS = {
    "hh": {"loader": hh_rows, "dataset": "Anthropic/hh-rlhf", "train_split": "train", "val_split": "test"},
    "uf": {"loader": uf_rows, "dataset": "HuggingFaceH4/ultrafeedback_binarized",
           "train_split": "train_prefs", "val_split": "test_prefs"},
}


def sample(rows, n, tokenizer, seed):
    """Shuffle with a fixed seed, keep n pairs, and cap the prompt at MAX_PROMPT_LENGTH tokens."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} valid single-turn pairs, need {n}")
    prompts, chosen, rejected, n_trunc = [], [], [], 0
    for instruction, c, r in rows[:n]:
        short = truncate_instruction(tokenizer, instruction, MAX_PROMPT_LENGTH)
        n_trunc += short != instruction
        prompts.append(format_prompt(short))
        chosen.append(c)
        rejected.append(r)
    return prompts, chosen, rejected, n_trunc


@torch.no_grad()
def evaluate(policy, reference, tokenizer, data, device):
    """Validation DPO loss, reward accuracy (strict r_w > r_l), reward margin."""
    policy.eval()
    prompts, chosen, rejected = data
    cr_all, rr_all, losses = [], [], []
    for i in range(0, len(prompts), EVAL_BATCH_SIZE):
        sl = slice(i, i + EVAL_BATCH_SIZE)
        with autocast_ctx(device):
            loss, cr, rr = core.dpo_forward(policy, reference, tokenizer, prompts[sl], chosen[sl],
                                            rejected[sl], BETA, MAX_LENGTH)
        cr_all.append(cr.float().cpu())
        rr_all.append(rr.float().cpu())
        losses.append(loss.item() * len(cr))
    policy.train()
    cr, rr = torch.cat(cr_all), torch.cat(rr_all)
    return {"val_loss": sum(losses) / len(cr), "val_reward_acc": (cr > rr).float().mean().item(),
            "val_reward_margin": (cr - rr).mean().item(), "val_chosen_reward": cr.mean().item(),
            "val_rejected_reward": rr.mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    track = TRACKS[parser.parse_args().track]
    name = "dpo_" + parser.parse_args().track
    run_dir = os.path.join(OUTPUTS, name)
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        print(f"{name} already finished ({run_dir}/summary.json exists); skipping.")
        return
    require_checkpoint(SFT_BEST_CHECKPOINT, "SFT best checkpoint")
    set_seed(SEED)
    policy_device, ref_device = get_devices()
    tokenizer = load_tokenizer()

    print(f"Loading {track['dataset']} ...")
    train = sample(track["loader"](track["train_split"]), N_TRAIN, tokenizer, SEED)
    val = sample(track["loader"](track["val_split"]), N_VAL, tokenizer, SEED)
    print(f"train pairs={len(train[0])} (prompts truncated: {train[3]}) | "
          f"val pairs={len(val[0])} (prompts truncated: {val[3]})")
    train_data, val_data = train[:3], val[:3]

    policy = AutoModelForCausalLM.from_pretrained(SFT_BEST_CHECKPOINT, torch_dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    policy.train()
    reference = AutoModelForCausalLM.from_pretrained(SFT_BEST_CHECKPOINT, torch_dtype=torch.float32).to(ref_device)
    reference.config.use_cache = False
    reference.eval()
    reference.requires_grad_(False)

    order = []
    for epoch in range(EPOCHS):
        idx = list(range(len(train_data[0])))
        random.Random(SEED + epoch).shuffle(idx)
        order += idx
    micro_batches = [order[i:i + BATCH_SIZE] for i in range(0, len(order), BATCH_SIZE)]
    windows = [micro_batches[i:i + GRAD_ACCUM] for i in range(0, len(micro_batches), GRAD_ACCUM)]
    total_steps = len(windows)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp(policy_device))

    config = {
        "stage": name, "dataset": track["dataset"], "train_split": track["train_split"],
        "val_split": track["val_split"],
        "filtering": "single-turn only (multi-turn dropped), non-empty, chosen != rejected; shuffled with seed",
        "n_train": len(train_data[0]), "n_val": len(val_data[0]),
        "n_train_prompts_truncated": train[3], "n_val_prompts_truncated": val[3],
        "initial_policy": SFT_BEST_CHECKPOINT, "reference_model": SFT_BEST_CHECKPOINT + " (frozen, no_grad)",
        "prompt_template": format_prompt("{instruction}"), "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH, "per_device_batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM, "optimizer": "AdamW (betas 0.9/0.999, eps 1e-8)",
        "lr": LR, "scheduler": "constant (no warmup, no decay)", "weight_decay": 0.0, "beta": BETA,
        "max_grad_norm": MAX_GRAD_NORM, "epochs": EPOCHS, "optimizer_steps": total_steps,
        "precision": ("policy fp32 weights + fp16 autocast + GradScaler; reference fp32 weights under the "
                      "same fp16 autocast") if use_amp(policy_device) else "fp32 (cpu)",
        "policy_device": str(policy_device), "reference_device": str(ref_device),
        "eval_every_steps": EVAL_EVERY, "checkpoint_selection": "highest validation reward accuracy",
        "trainable": "full fine-tuning", "seed": SEED,
    }
    write_json(os.path.join(run_dir, "config.json"), config)
    print(config)

    log = MetricsLogger(os.path.join(run_dir, "metrics.jsonl"))
    best_dir, final_dir = os.path.join(run_dir, "best"), os.path.join(run_dir, "final")
    best = {"val_reward_acc": -1.0, "step": None}

    def run_eval(step):
        m = evaluate(policy, reference, tokenizer, val_data, policy_device)
        log.log({"type": "eval", "step": step, **m})
        msg = (f"  [eval] step {step}: loss={m['val_loss']:.4f} acc={m['val_reward_acc']:.3f} "
               f"margin={m['val_reward_margin']:.4f}")
        if step > 0 and m["val_reward_acc"] > best["val_reward_acc"]:
            best.update(val_reward_acc=m["val_reward_acc"], step=step, metrics=m)
            save_checkpoint(policy, tokenizer, best_dir)
            msg += "  -> new best, saved"
        print(msg, flush=True)
        return m

    run_eval(0)  # policy == reference here, so every reward pair is tied (acc 0, margin 0, loss log 2)
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, window in enumerate(windows, start=1):
        losses, crs, rrs = [], [], []
        for mb in window:
            with autocast_ctx(policy_device):
                loss, cr, rr = core.dpo_forward(policy, reference, tokenizer,
                                                [train_data[0][j] for j in mb], [train_data[1][j] for j in mb],
                                                [train_data[2][j] for j in mb], BETA, MAX_LENGTH)
            scaler.scale(loss / len(window)).backward()
            losses.append(loss.item())
            crs.append(cr.float().cpu())
            rrs.append(rr.float().cpu())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        cr, rr = torch.cat(crs), torch.cat(rrs)
        rec = {"type": "train", "step": step, "train_loss": sum(losses) / len(losses),
               "train_reward_acc": (cr > rr).float().mean().item(),
               "train_reward_margin": (cr - rr).mean().item(), "grad_norm": grad_norm,
               "loss_scale": scaler.get_scale(), "elapsed_s": time.time() - start}
        log.log(rec)
        if step % 10 == 0 or step == 1:
            eta = (time.time() - start) / step * (total_steps - step) / 60
            print(f"step {step}/{total_steps} | loss {rec['train_loss']:.4f} | acc {rec['train_reward_acc']:.3f} | "
                  f"margin {rec['train_reward_margin']:.4f} | grad_norm {grad_norm:.2f} | eta {eta:.0f} min", flush=True)
        if step % EVAL_EVERY == 0 and step != total_steps:
            run_eval(step)

    final = run_eval(total_steps)
    save_checkpoint(policy, tokenizer, final_dir)
    summary = {"best_step": best["step"], "best_val_reward_acc": best["val_reward_acc"],
               "best_metrics": best.get("metrics"), "final_step": total_steps, "final_metrics": final,
               "best_checkpoint": best_dir, "final_checkpoint": final_dir,
               "reference_checkpoint": SFT_BEST_CHECKPOINT, "train_minutes": (time.time() - start) / 60}
    write_json(os.path.join(run_dir, "summary.json"), summary)
    print(f"{name} done:", summary)


if __name__ == "__main__":
    main()
