"""Stage 1 / Task 3: Supervised Fine-Tuning (SFT)."""
import os, sys, math
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset

sys.path.insert(0, ".")
import submission_adapter as adapter
from common import format_prompt, save_checkpoint, robust_load_dataset, require_checkpoint

# ======== PARAMS (full run per manual) ========
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
N_TRAIN = 30000
N_VAL = 1000
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 16
LR = 2e-5
WARMUP_RATIO = 0.03
MAX_STEPS = None
EVAL_EVERY = 100
CHECKPOINT_ROOT = "outputs/sft/full"
# ======================================================

def evaluate(model, tokenizer, val_prompts, val_responses):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(val_prompts), BATCH_SIZE):
            bp = val_prompts[i:i+BATCH_SIZE]
            br = val_responses[i:i+BATCH_SIZE]
            if not bp:
                continue
            vloss = adapter.compute_sft_loss(model, tokenizer, bp, br, MAX_LENGTH)
            losses.append(vloss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

def main():
    best_dir = os.path.join(CHECKPOINT_ROOT, "best")
    final_dir = os.path.join(CHECKPOINT_ROOT, "final")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to("cuda:0")
    model.train()
    print("model and tokenizer loaded")

    print("Loading UltraChat slice...")
    raw_train = robust_load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N_TRAIN}]")
    raw_val = robust_load_dataset("HuggingFaceH4/ultrachat_200k", split=f"test_sft[:{N_VAL}]")

    def extract(example):
        msgs = example["messages"]
        instr = next((m["content"] for m in msgs if m["role"] == "user"), None)
        resp = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        return instr, resp

    def build_pairs(ds):
        prompts, responses = [], []
        for ex in ds:
            instr, resp = extract(ex)
            if instr and resp:
                prompts.append(format_prompt(instr))
                responses.append(resp)
        return prompts, responses

    train_prompts, train_responses = build_pairs(raw_train)
    val_prompts, val_responses = build_pairs(raw_val)
    print(f"Train pairs: {len(train_prompts)} | Val pairs: {len(val_prompts)}")

    num_micro_batches = math.ceil(len(train_prompts) / BATCH_SIZE)
    total_optim_steps = MAX_STEPS if MAX_STEPS is not None else math.ceil(num_micro_batches / GRAD_ACCUM)
    warmup_steps = max(1, int(total_optim_steps * WARMUP_RATIO))

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps)
    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for i in range(0, len(train_prompts), BATCH_SIZE):
        if MAX_STEPS is not None and step >= MAX_STEPS:
            break
        bp = train_prompts[i:i+BATCH_SIZE]
        br = train_responses[i:i+BATCH_SIZE]
        if not bp:
            continue

        with autocast("cuda", dtype=torch.float16):
            loss = adapter.compute_sft_loss(model, tokenizer, bp, br, MAX_LENGTH)
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (i // BATCH_SIZE + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step} | loss: {loss.item() * GRAD_ACCUM:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}", flush=True)

            if step % EVAL_EVERY == 0 or (MAX_STEPS is not None and step == MAX_STEPS):
                val_loss = evaluate(model, tokenizer, val_prompts, val_responses)
                print(f"  -> val loss at step {step}: {val_loss:.4f}", flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, tokenizer, best_dir)
                    print(f"  -> new best (val loss {val_loss:.4f}), saved to {best_dir}", flush=True)

    final_val_loss = evaluate(model, tokenizer, val_prompts, val_responses)
    print(f"Final val loss: {final_val_loss:.4f}")
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        save_checkpoint(model, tokenizer, best_dir)
        print(f"Final step is also the best checkpoint, saved to {best_dir}")

    save_checkpoint(model, tokenizer, final_dir)
    print(f"Saved final checkpoint to {final_dir}")
    print(f"Best val loss observed: {best_val_loss:.4f} (saved at {best_dir})")

if __name__ == "__main__":
    main()
