"""Core SFT / DPO math shared by the training scripts and submission_adapter.py.

Conventions (documented in the report):
- Every response gets the tokenizer's EOS token appended *before* truncation to
  max_length, so the EOS can be cut off when a response is truncated.
- Sequences are right-padded. Prompt tokens and padding tokens are labelled -100.
- A sequence score is the SUM of response-token log-probabilities.
- SFT loss reduction: mean negative log-likelihood over all unmasked response
  tokens in the batch (token-level mean).
- DPO loss: batch mean of -log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l))).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _input_ids(encoded):
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded.input_ids)


def build_lm_batch(tokenizer, prompts, responses, max_length):
    """Tokenize prompt+response rows into a right-padded LM batch whose labels
    are -100 everywhere except on response tokens (EOS included)."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    rows_ids, rows_labels = [], []
    for prompt, response in zip(prompts, responses):
        prompt_ids = _input_ids(tokenizer(prompt, add_special_tokens=False))
        response_ids = _input_ids(tokenizer(response, add_special_tokens=False))
        if eos_id is not None:
            response_ids = response_ids + [eos_id]
        full_ids = (prompt_ids + response_ids)[:max_length]
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = [IGNORE_INDEX] * n_prompt + full_ids[n_prompt:]
        rows_ids.append(full_ids)
        rows_labels.append(labels)

    width = max(len(ids) for ids in rows_ids)
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in zip(rows_ids, rows_labels):
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def response_token_logps(model, batch):
    """Per-token log p(label) on response positions (0 elsewhere) and the mask.
    Returns (token_logps [B, T-1], mask [B, T-1]) on the model's device."""
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~mask, 0)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    return token_logps, mask


def sequence_logps(model, batch):
    """Summed response-token log-probability, one value per row (row order kept)."""
    token_logps, _ = response_token_logps(model, batch)
    return token_logps.sum(dim=1)


def sft_loss(model, tokenizer, prompts, responses, max_length):
    """Response-only SFT loss: mean NLL over all unmasked response tokens."""
    batch = build_lm_batch(tokenizer, prompts, responses, max_length)
    token_logps, mask = response_token_logps(model, batch)
    n_tokens = mask.sum().clamp(min=1)
    return -token_logps.sum() / n_tokens


def dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                        reference_chosen_logps, reference_rejected_logps, beta):
    """Batch-mean DPO loss plus the implicit rewards used for diagnostics."""
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return losses.mean(), chosen_rewards, rejected_rewards


def dpo_forward(policy_model, reference_model, tokenizer, prompts,
                chosen_responses, rejected_responses, beta, max_length):
    """Full DPO forward pass. Gradients flow only through the policy; the frozen
    reference is evaluated under torch.no_grad() on its own device.
    Returns (loss, chosen_rewards, rejected_rewards); rewards are detached."""
    chosen_batch = build_lm_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_lm_batch(tokenizer, prompts, rejected_responses, max_length)
    policy_device = next(policy_model.parameters()).device

    policy_chosen = sequence_logps(policy_model, chosen_batch)
    policy_rejected = sequence_logps(policy_model, rejected_batch)
    with torch.no_grad():
        reference_chosen = sequence_logps(reference_model, chosen_batch).to(policy_device)
        reference_rejected = sequence_logps(reference_model, rejected_batch).to(policy_device)

    loss, chosen_rewards, rejected_rewards = dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    return loss, chosen_rewards.detach(), rejected_rewards.detach()
