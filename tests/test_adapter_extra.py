"""Extra edge-case checks beyond the public smoke tests (manual section 4.2):
truncation boundaries, mixed lengths, row order, exact numerics, small steps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def _manual_logp(model, tokenizer, prompt, response, max_length):
    ids = tokenizer.encode(prompt) + tokenizer.encode(response) + [tokenizer.eos_token_id]
    ids = ids[:max_length]
    n_prompt = min(len(tokenizer.encode(prompt)), len(ids))
    logits = model(torch.tensor([ids])).logits[0]
    logp = F.log_softmax(logits, dim=-1)
    return sum(logp[t - 1, ids[t]].item() for t in range(max(n_prompt, 1), len(ids)))


def test_truncation_drops_eos_and_respects_max_length():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: "], ["abcdefghij"], max_length=8)
    assert batch["input_ids"].shape == (1, 8)
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    assert batch["labels"][0, :4].eq(-100).all() and batch["labels"][0, 4:].ne(-100).all()


def test_prompt_longer_than_max_length_has_no_response_tokens():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["abcdefghij", "a"], ["xy", "xy"], max_length=6)
    assert batch["labels"][0].eq(-100).all()
    assert batch["labels"][1].ne(-100).sum() == 3  # "x", "y", EOS


def test_row_order_mixed_lengths_and_exact_logps():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts = ["long prompt here: ", "q: ", "mid one: "]
    responses = ["a", "a much longer answer", "mid"]
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=24)
    expected = [_manual_logp(model, tok, p, r, 24) for p, r in zip(prompts, responses)]
    torch.testing.assert_close(logps, torch.tensor(expected), rtol=1e-5, atol=1e-5)
    rev = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], max_length=24)
    torch.testing.assert_close(rev, logps.flip(0))


def test_padding_does_not_change_scores():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    alone = adapter.compute_response_logprobs(model, tok, ["p: "], ["ok"], max_length=32)
    padded = adapter.compute_response_logprobs(model, tok, ["p: ", "p: "], ["ok", "a long reply"], max_length=32)
    torch.testing.assert_close(alone[0], padded[0])


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    prompts, responses = ["a: ", "b: "], ["x", "longer"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=16)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=16)
    n_tokens = (1 + 1) + (6 + 1)
    torch.testing.assert_close(loss, -logps.sum() / n_tokens)


def test_dpo_formula_and_reference_is_frozen():
    pc, pr = torch.tensor([-1.0, -3.0]), torch.tensor([-2.0, -1.0])
    rc, rr = torch.tensor([-1.5, -2.0]), torch.tensor([-1.5, -2.0])
    beta = 0.1
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    z = beta * ((pc - pr) - (rc - rr))
    expected = torch.tensor([math.log1p(math.exp(-v)) for v in z.tolist()]).mean()
    torch.testing.assert_close(loss, expected)

    tok = ToyTokenizer()
    policy, reference = ToyCausalLM(tok.vocab_size, 0.2), ToyCausalLM(tok.vocab_size, 0.2)
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["P: "], ["good"], ["bad"], 0.1, 16)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))  # identical models -> log 2
    loss.backward()
    assert reference.logit_bias.grad is None


def test_small_dpo_step_increases_preference():
    tok = ToyTokenizer()
    policy, reference = ToyCausalLM(tok.vocab_size, 0.2), ToyCausalLM(tok.vocab_size, 0.2)
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    args = (tok, ["P: "], ["aaa"], ["zzz"], 0.5, 16)
    before = adapter.compute_dpo_loss(policy, reference, *args)
    before.backward()
    opt.step()
    after = adapter.compute_dpo_loss(policy, reference, *args)
    assert after.item() < before.item()
