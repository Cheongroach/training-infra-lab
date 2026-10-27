"""Synchronous on-policy sampling; every rank executes the same decode count."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from .contracts import Trajectory
from .executor import synchronize
from .rewards import STRICT_RULE, instruction, score_answer


@torch.no_grad()
def _sample_group(actor, tokenizer, row, *, prompt_id, policy_version, group_size,
                 device, max_new_tokens=256, autocast_dtype=torch.bfloat16, reward_rule=STRICT_RULE):
    """Temperature=1, no top-k/top-p: recorded behavior probabilities are exact.

    Actor root owns both embedding and head, including during cached decode.
    Finished ranks keep participating until global completion; finished rows'
    extra computation is excluded from trajectory masks and reported as waste.
    """
    if group_size < 2 or max_new_tokens < 1:
        raise ValueError("Invalid sampling configuration")
    messages = [{"role": "user", "content": row["question"] +
                 "\n" + instruction(reward_rule)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=False,
    )
    ids = torch.tensor([prompt] * group_size, device=device)
    attention = torch.ones_like(ids)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer EOS is required")
    finished = [False] * group_size
    tokens, logs = [[] for _ in finished], [[] for _ in finished]
    cache, steps = None, 0
    previous_training = actor.training
    actor.eval()
    synchronize(device)
    start = time.perf_counter()
    try:
        for _ in range(max_new_tokens):
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits, cache = actor(ids, attention, generation=True, past_key_values=cache)
            log_probs = torch.log_softmax(logits.float(), -1)
            sampled = torch.multinomial(log_probs.exp(), 1)
            chosen = log_probs.gather(1, sampled).squeeze(1).cpu().tolist()
            selected = sampled.squeeze(1).cpu().tolist()
            for i, token in enumerate(selected):
                if not finished[i]:
                    tokens[i].append(token)
                    logs[i].append(chosen[i])
                    finished[i] = token == eos
            steps += 1
            active = torch.tensor(int(not all(finished)), device=device)
            if dist.is_initialized():
                dist.all_reduce(active, op=dist.ReduceOp.MAX)
            if active.item() == 0:
                break
            ids = sampled
            attention = torch.cat([attention, torch.ones_like(sampled)], dim=1)
        synchronize(device)
        duration = time.perf_counter() - start
        trajectories, details = [], []
        for i, (generated, log_probs) in enumerate(zip(tokens, logs)):
            text = tokenizer.decode(generated, skip_special_tokens=True)
            finish_reason = "eos" if finished[i] else "length"
            scored = score_answer(text, row["answer"], finish_reason, reward_rule)
            reward = scored["reward"]
            trajectory_id = f"v{policy_version}/{prompt_id}/{i}"
            trajectories.append(Trajectory(
                trajectory_id, prompt_id, policy_version, tuple(prompt + generated),
                (0,) * (len(prompt) - 1) + (1,) * len(generated),
                (0.0,) * (len(prompt) - 1) + tuple(log_probs), reward,
            ))
            details.append({
                "trajectory_id": trajectory_id, "text": text, "reward": reward,
                "finish_reason": finish_reason, **scored,
                "prompt_tokens": len(prompt), "response_tokens": len(generated),
            })
        return trajectories, {
            "seconds": duration, "decode_steps": steps,
            "wasted_decode_rows": steps * group_size - sum(map(len, tokens)),
            "sampling": {"temperature": 1.0, "top_k": None, "top_p": None},
            "reward_rule": reward_rule, "trajectories": details,
        }
    finally:
        actor.train(previous_training)


@torch.no_grad()
def sample_group(actor, tokenizer, row, *, prompt_id, policy_version, group_size,
                 device, max_new_tokens=256, autocast_dtype=torch.bfloat16, resident=False,
                 reward_rule=STRICT_RULE):
    from .residency import rollout_residency

    synchronize(device)
    start = time.perf_counter()
    with rollout_residency(actor, device, enabled=resident) as decision:
        trajectories, details = _sample_group(
            actor, tokenizer, row, prompt_id=prompt_id, policy_version=policy_version,
            group_size=group_size, device=device, max_new_tokens=max_new_tokens,
            autocast_dtype=autocast_dtype, reward_rule=reward_rule)
    synchronize(device)
    details.update(residency=decision, phase_seconds=time.perf_counter() - start)
    return trajectories, details
