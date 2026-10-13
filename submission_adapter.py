"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

def _format_prompt(prompt):
    return prompt
def _get_input_ids(encoded):
    if isinstance(encoded, dict):
        return encoded["input_ids"]
    return encoded.input_ids

def _build_single_batch(tokenizer, prompts, responses, max_length):
    input_ids_list = []
    labels_list = []
    attn_list = []

    eos_id = tokenizer.eos_token_id

    for prompt, response in zip(prompts, responses):
        prompt_text = _format_prompt(prompt)
        prompt_ids = _get_input_ids(tokenizer(prompt_text, add_special_tokens=False))
        response_ids = _get_input_ids(tokenizer(response, add_special_tokens=False))

        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        full_ids = prompt_ids + response_ids
        full_ids = full_ids[:max_length]

        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = [-100] * n_prompt + full_ids[n_prompt:]
        labels = labels[:max_length]

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append([1] * len(full_ids))

    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    padded_input_ids = []
    padded_labels = []
    padded_attn = []

    for ids, labs, attn in zip(input_ids_list, labels_list, attn_list):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_labels.append(labs + [-100] * pad_len)
        padded_attn.append(attn + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def _sum_logps_from_batch(model, batch, device, no_grad):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        mask = (shift_labels != -100)
        safe_labels = shift_labels.clone()
        safe_labels[~mask] = 0

        token_logps = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * mask
        return token_logps.sum(dim=1)

def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return _build_single_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    return _sum_logps_from_batch(model, batch, device, no_grad=False)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    mask = (shift_labels != -100)
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    token_logps = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_nll = -token_logps * mask

    total_nll = token_nll.sum()
    total_tokens = mask.sum().clamp(min=1)
    loss = total_nll / total_tokens
    return loss


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    chosen_batch = _build_single_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = _build_single_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_gap - reference_gap)
    loss = -F.logsigmoid(logits)
    return loss.mean()

def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    batches = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    device = next(policy_model.parameters()).device

    policy_chosen_logps = _sum_logps_from_batch(policy_model, batches["chosen"], device, no_grad=False)
    policy_rejected_logps = _sum_logps_from_batch(policy_model, batches["rejected"], device, no_grad=False)

    ref_device = next(reference_model.parameters()).device
    ref_chosen_batch = {k: v.to(ref_device) if torch.is_tensor(v) else v for k, v in batches["chosen"].items()}
    ref_rejected_batch = {k: v.to(ref_device) if torch.is_tensor(v) else v for k, v in batches["rejected"].items()}

    reference_chosen_logps = _sum_logps_from_batch(reference_model, ref_chosen_batch, ref_device, no_grad=True)
    reference_rejected_logps = _sum_logps_from_batch(reference_model, ref_rejected_batch, ref_device, no_grad=True)

    reference_chosen_logps = reference_chosen_logps.to(device)
    reference_rejected_logps = reference_rejected_logps.to(device)

    return compute_dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps, beta,
    )
