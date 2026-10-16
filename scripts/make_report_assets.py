"""Task 8 support: training plots and report tables built from the saved run
artifacts (metrics.jsonl / config.json / summary.json / log_history.json)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment.common import GENERATION_SETTINGS, OUTPUTS, format_prompt, read_json, read_jsonl

REPORT_DIR = os.path.join(OUTPUTS, "report")
EMA = 0.9


def ema(values, alpha=EMA):
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def split_metrics(run):
    rows = read_jsonl(os.path.join(OUTPUTS, run, "metrics.jsonl"))
    return [r for r in rows if r["type"] == "train"], [r for r in rows if r["type"] == "eval"]


def plot_sft():
    train, evals = split_metrics("sft")
    summary = read_json(os.path.join(OUTPUTS, "sft", "summary.json"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = [r["step"] for r in train]
    ax.plot(steps, [r["train_loss"] for r in train], color="#9ecae1", lw=0.8, label="train loss (per step)")
    ax.plot(steps, ema([r["train_loss"] for r in train]), color="#08519c", lw=1.8, label=f"train loss (EMA {EMA})")
    ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], "o-", color="#d94801", label="validation loss")
    ax.axvline(summary["best_step"], color="gray", ls="--", lw=1, label=f"best checkpoint (step {summary['best_step']})")
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token NLL (mean per token)")
    ax.set_title("SFT: Qwen2.5-0.5B on UltraChat (29,997 train / 1,000 val)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "sft_loss.png"), dpi=150)
    plt.close(fig)


def plot_dpo(run, title):
    train, evals = split_metrics(run)
    summary = read_json(os.path.join(OUTPUTS, run, "summary.json"))
    best = summary["best_step"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    steps, esteps = [r["step"] for r in train], [r["step"] for r in evals]

    ax = axes[0]
    ax.plot(steps, [r["train_loss"] for r in train], color="#9ecae1", lw=0.8, label="train loss (per step)")
    ax.plot(steps, ema([r["train_loss"] for r in train]), color="#08519c", lw=1.8, label=f"train loss (EMA {EMA})")
    ax.plot(esteps, [r["val_loss"] for r in evals], "o-", color="#d94801", label="validation loss")
    ax.set_ylabel("DPO loss")
    ax.set_title("DPO loss")

    ax = axes[1]
    ax.plot(steps, ema([r["train_reward_acc"] for r in train]), color="#08519c", label=f"train (EMA {EMA})")
    ax.plot(esteps, [r["val_reward_acc"] for r in evals], "o-", color="#d94801", label="validation")
    ax.set_ylabel("reward accuracy  P(r_w > r_l)")
    ax.set_title("Reward accuracy")

    ax = axes[2]
    ax.plot(steps, ema([r["train_reward_margin"] for r in train]), color="#08519c", label=f"train (EMA {EMA})")
    ax.plot(esteps, [r["val_reward_margin"] for r in evals], "o-", color="#d94801", label="validation")
    ax.set_ylabel("reward margin  mean(r_w - r_l)")
    ax.set_title("Reward margin")

    for ax in axes:
        if best is not None:
            ax.axvline(best, color="gray", ls="--", lw=1, label=f"best checkpoint (step {best})")
        ax.set_xlabel("optimizer step (effective batch 32)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, f"{run}_training.png"), dpi=150)
    plt.close(fig)


def plot_lora():
    hist = read_json(os.path.join(OUTPUTS, "lora_dpo", "log_history.json"))
    train = [h for h in hist if "loss" in h and "rewards/accuracies" in h]
    evals = [h for h in hist if "eval_loss" in h and "step" in h]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    s = [h["step"] for h in train]
    es = [h["step"] for h in evals]
    pairs = [("loss", "eval_loss", "DPO loss"), ("rewards/accuracies", "eval_rewards/accuracies", "reward accuracy"),
             ("rewards/margins", "eval_rewards/margins", "reward margin")]
    for ax, (tk, ek, label) in zip(axes, pairs):
        ax.plot(s, [h[tk] for h in train], color="#9ecae1", lw=0.8, label="train (per step)")
        ax.plot(s, ema([h[tk] for h in train]), color="#08519c", lw=1.8, label=f"train (EMA {EMA})")
        if evals and ek in evals[0]:
            ax.plot(es, [h[ek] for h in evals], "o-", color="#d94801", label="held-out (20 pairs)")
        ax.set_xlabel("optimizer step (effective batch 8)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("LoRA-DPO (TRL DPOTrainer): SmolLM2-360M-Instruct on the Ramsay persona pairs")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "lora_dpo_training.png"), dpi=150)
    plt.close(fig)


def config_table():
    sft = read_json(os.path.join(OUTPUTS, "sft", "config.json"))
    hh = read_json(os.path.join(OUTPUTS, "dpo_hh", "config.json"))
    uf = read_json(os.path.join(OUTPUTS, "dpo_uf", "config.json"))
    lora = read_json(os.path.join(OUTPUTS, "lora_dpo", "config.json"))

    def full_row(name, c, beta="-", ref="-"):
        return [name, "full fine-tuning", c["optimizer"], f"{c['lr']:g}", c["scheduler"],
                str(c.get("warmup_ratio", 0)), str(c["weight_decay"]), str(c["max_grad_norm"]),
                str(c["per_device_batch_size"]), str(c["grad_accum"]), str(c["effective_batch_size"]),
                c["precision"], f"{c['max_length']}" + (f" (prompt {c['max_prompt_length']})" if c.get("max_prompt_length") else ""),
                f"{c['epochs']} ep / {c['optimizer_steps']} steps", str(c["seed"]), str(beta), ref,
                f"{c['n_train']} / {c['n_val']}"]

    rows = [
        full_row("SFT", sft),
        full_row("HH-RLHF DPO", hh, hh["beta"], hh["reference_model"]),
        full_row("UltraFeedback DPO", uf, uf["beta"], uf["reference_model"]),
        ["LoRA-DPO", f"LoRA r={lora['lora_r']}, alpha={lora['lora_alpha']}, dropout={lora['lora_dropout']}, "
                     f"targets={','.join(lora['target_modules'])} ({lora['trainable_params']:,} trainable)",
         lora["optimizer"], f"{lora['lr']:g}", lora["scheduler"], str(lora["warmup_ratio"]),
         str(lora["weight_decay"]), str(lora["max_grad_norm"]), str(lora["per_device_batch_size"]),
         str(lora["grad_accum"]), str(lora["effective_batch_size"]), lora["precision"],
         f"{lora['max_length']} (prompt {lora['max_prompt_length']})",
         f"{lora['epochs']} ep / {lora['optimizer_steps']} steps", str(lora["seed"]), str(lora["beta"]),
         lora["reference_model"], f"{lora['n_train']} / {lora['n_heldout']}"],
    ]
    header = ["Run", "Trainable params", "Optimizer", "Peak LR", "Scheduler", "Warmup", "Weight decay",
              "Grad clip", "Per-device batch", "Grad accum", "Effective batch", "Precision", "Max seq len",
              "Epochs / steps", "Seed", "DPO beta", "Reference model", "Train / val examples"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    gen = read_json(os.path.join(OUTPUTS, "eval", "generation_settings.json"))
    lines += ["", "**Generation settings (all models, all evaluation sets):** "
              + ", ".join(f"{k}={v}" for k, v in GENERATION_SETTINGS.items())
              + "; temperature/top-p not used (greedy); batch size "
              + f"{gen['batch_size']}, left padding.",
              f"Qwen prompt template: `{format_prompt('{instruction}')!r}` (no chat template). "
              "SmolLM2 models use their own chat template with add_generation_prompt=True."]
    return "\n".join(lines)


def training_table():
    lines = ["| Run | Selected checkpoint | Selection metric (best) | Final-step metric | Train time (min) |",
             "|---|---|---|---|---|"]
    s = read_json(os.path.join(OUTPUTS, "sft", "summary.json"))
    mins = "-" if s.get("train_minutes") is None else f"{s['train_minutes']:.0f}"
    lines.append(f"| SFT | outputs/sft/best (step {s['best_step']}) | val loss {s['best_val_loss']:.4f} | "
                 f"val loss {s['final_val_loss']:.4f} (step {s['final_step']}) | {mins} |")
    for run, label in [("dpo_hh", "HH-RLHF DPO"), ("dpo_uf", "UltraFeedback DPO")]:
        s = read_json(os.path.join(OUTPUTS, run, "summary.json"))
        b, f = s["best_metrics"], s["final_metrics"]
        lines.append(f"| {label} | outputs/{run}/best (step {s['best_step']}) | reward acc {b['val_reward_acc']:.3f}, "
                     f"margin {b['val_reward_margin']:.4f}, val loss {b['val_loss']:.4f} | reward acc "
                     f"{f['val_reward_acc']:.3f}, margin {f['val_reward_margin']:.4f}, val loss {f['val_loss']:.4f} "
                     f"(step {s['final_step']}) | {s['train_minutes']:.0f} |")
    s = read_json(os.path.join(OUTPUTS, "lora_dpo", "summary.json"))
    b, a = s["heldout_before"], s["heldout_after"]
    lines.append(f"| LoRA-DPO | outputs/lora_dpo/adapter (end of training) | held-out reward acc "
                 f"{a.get('eval_rewards/accuracies', float('nan')):.3f}, margin {a.get('eval_rewards/margins', float('nan')):.3f}, "
                 f"loss {a.get('eval_loss', float('nan')):.4f} | before training: loss {b.get('eval_loss', float('nan')):.4f} | - |")
    return "\n".join(lines)


def data_table():
    sft = read_json(os.path.join(OUTPUTS, "sft", "config.json"))
    hh = read_json(os.path.join(OUTPUTS, "dpo_hh", "config.json"))
    uf = read_json(os.path.join(OUTPUTS, "dpo_uf", "config.json"))
    lora = read_json(os.path.join(OUTPUTS, "lora_dpo", "config.json"))
    return "\n".join([
        "| Track | Source | Train | Val | Filtering / formatting | Evaluation set |", "|---|---|---|---|---|---|",
        f"| SFT | {sft['train_data']} | {sft['n_train']} | {sft['n_val']} (test_sft) | {sft['filtering']} | AlpacaEval-300 (seed 42) |",
        f"| HH-RLHF DPO | {hh['dataset']} ({hh['train_split']}/{hh['val_split']}) | {hh['n_train']} | {hh['n_val']} | "
        f"{hh['filtering']}; {hh['n_train_prompts_truncated']} train prompts truncated | eval_sets/hh_rlhf_eval_300_seed42.json (300) |",
        f"| UltraFeedback DPO | {uf['dataset']} ({uf['train_split']}/{uf['val_split']}) | {uf['n_train']} | {uf['n_val']} | "
        f"{uf['filtering']}; {uf['n_train_prompts_truncated']} train prompts truncated | AlpacaEval-300 (seed 42) |",
        f"| LoRA-DPO | {lora['data']} | {lora['n_train']} | {lora['n_heldout']} (held out) | {lora['format']} | "
        "outputs/eval/persona_style_eval_prompts.json (24) |",
    ])


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    if os.path.exists(os.path.join(OUTPUTS, "sft", "metrics.jsonl")):
        plot_sft()
    plot_dpo("dpo_hh", "From-scratch DPO on HH-RLHF (15k pairs, beta 0.1, lr 1e-6)")
    plot_dpo("dpo_uf", "From-scratch DPO on UltraFeedback (15k pairs, beta 0.1, lr 1e-6)")
    plot_lora()
    md = "\n\n".join(["## Data", data_table(), "## Experimental configuration", config_table(),
                      "## Training results", training_table()])
    with open(os.path.join(REPORT_DIR, "report_tables.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print("\nplots:", sorted(p for p in os.listdir(REPORT_DIR) if p.endswith(".png")))


if __name__ == "__main__":
    main()
