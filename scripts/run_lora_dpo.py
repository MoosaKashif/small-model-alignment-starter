"""Task 5: TRL DPOTrainer + PEFT/LoRA persona tuning of SmolLM2-360M-Instruct on
data/persona_lora/ramsay_style_preferences.json (the only stage that uses TRL)."""
import os

# HF Trainer wraps the model in nn.DataParallel when several GPUs are visible,
# which DPOTrainer does not support; pin this process to one GPU before torch loads.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import inspect
import json
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.common import OUTPUTS, SEED, apply_overrides, count_params, set_seed, write_json

# ---------------- configuration ----------------
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_PATH = "data/persona_lora/ramsay_style_preferences.json"
N_HELDOUT = 20                 # held-out pairs for eval loss / reward accuracy
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LR = 5e-5
BATCH_SIZE = 4
GRAD_ACCUM = 2
EPOCHS = 3
BETA = 0.1
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256
WARMUP_RATIO = 0.1
EVAL_STEPS = 5
RUN_DIR = os.path.join(OUTPUTS, "lora_dpo")
# ------------------------------------------------
apply_overrides(globals(), "lora")


def to_conversational(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main():
    adapter_dir = os.path.join(RUN_DIR, "adapter")
    if os.path.exists(os.path.join(RUN_DIR, "summary.json")):
        print(f"LoRA-DPO already finished ({RUN_DIR}/summary.json exists); skipping.")
        return
    set_seed(SEED)
    with open(DATA_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    heldout_idx, train_idx = sorted(idx[:N_HELDOUT]), sorted(idx[N_HELDOUT:])
    train_ds = Dataset.from_list([to_conversational(rows[i]) for i in train_idx])
    eval_ds = Dataset.from_list([to_conversational(rows[i]) for i in heldout_idx])
    print(f"persona pairs: total={len(rows)} train={len(train_ds)} heldout={len(eval_ds)}")
    print(train_ds[0])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 base weights; the trainable LoRA weights stay fp32 under fp16 AMP.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    peft_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                             target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")

    on_gpu = torch.cuda.is_available()
    wanted = dict(
        output_dir=os.path.join(RUN_DIR, "trainer_tmp"), per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS, beta=BETA, max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH,
        logging_steps=1, eval_strategy="steps", evaluation_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="no", report_to="none", fp16=on_gpu, bf16=False, seed=SEED,
        max_grad_norm=1.0, weight_decay=0.0, optim="adamw_torch", remove_unused_columns=False,
    )
    accepted = set(inspect.signature(DPOConfig.__init__).parameters)
    dropped = sorted(k for k in wanted if k not in accepted)
    if "eval_strategy" in accepted:
        dropped = sorted(set(dropped) | {"evaluation_strategy"})
    for key in dropped:
        wanted.pop(key, None)
    print(f"DPOConfig keys not supported by this TRL version (dropped): {dropped}")
    args = DPOConfig(**wanted)

    trainer_kwargs = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                          peft_config=peft_config)
    if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    total, trainable = count_params(trainer.model)
    print(f"parameters: total={total:,} trainable(LoRA)={trainable:,} ({100 * trainable / total:.2f}%)")

    initial_eval = trainer.evaluate()
    print("held-out metrics before training:", initial_eval)
    trainer.train()
    final_eval = trainer.evaluate()
    print("held-out metrics after training:", final_eval)

    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    write_json(os.path.join(RUN_DIR, "log_history.json"), trainer.state.log_history)
    import trl
    import peft
    config = {
        "stage": "lora_dpo", "base_model": MODEL_NAME, "data": DATA_PATH, "n_total": len(rows),
        "n_train": len(train_ds), "n_heldout": len(eval_ds), "heldout_indices": heldout_idx,
        "format": "TRL conversational (prompt/chosen/rejected as role-message lists); model chat template",
        "trainable": "LoRA", "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES, "trainable_params": trainable, "total_params": total,
        "reference_model": "base model with the LoRA adapter disabled (TRL default when peft_config is given)",
        "optimizer": "AdamW (adamw_torch)", "lr": LR, "scheduler": "cosine", "warmup_ratio": WARMUP_RATIO,
        "weight_decay": 0.0, "max_grad_norm": 1.0, "per_device_batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "effective_batch_size": BATCH_SIZE * GRAD_ACCUM, "n_gpus_used": 1,
        "epochs": EPOCHS, "optimizer_steps": trainer.state.global_step, "beta": BETA,
        "max_length": MAX_LENGTH, "max_prompt_length": MAX_PROMPT_LENGTH if "max_prompt_length" not in dropped else "n/a in this TRL",
        "precision": "fp32 base + fp16 AMP" if on_gpu else "fp32 (cpu)", "seed": SEED,
        "trl_version": trl.__version__, "peft_version": peft.__version__, "dropped_config_keys": dropped,
    }
    write_json(os.path.join(RUN_DIR, "config.json"), config)
    write_json(os.path.join(RUN_DIR, "summary.json"), {"heldout_before": initial_eval, "heldout_after": final_eval,
                                                       "adapter_dir": adapter_dir})
    print("LoRA-DPO done:", config)


if __name__ == "__main__":
    main()
