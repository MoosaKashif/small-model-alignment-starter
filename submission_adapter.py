"""Stable grading interface for the SFT and DPO implementation.

Each function is a thin wrapper around alignment/core.py, which is the same code
the SFT and from-scratch DPO training scripts use.
SFT loss reduction: mean over all unmasked response tokens in the batch.
"""

from __future__ import annotations

from alignment import core


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""
    return core.build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    batch = core.build_lm_batch(tokenizer, prompts, responses, max_length)
    return core.sequence_logps(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss (token-level mean reduction)."""
    return core.sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""
    return {
        "chosen": core.build_lm_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": core.build_lm_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""
    loss, _, _ = core.dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps, beta,
    )
    return loss


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
    """Return a scalar mean DPO loss with a frozen (no_grad) reference computation."""
    loss, _, _ = core.dpo_forward(
        policy_model, reference_model, tokenizer, prompts,
        chosen_responses, rejected_responses, beta, max_length,
    )
    return loss
