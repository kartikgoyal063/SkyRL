"""Self-distillation (SDPO) teacher-input construction — task-agnostic.

SDPO conditions a *teacher* (the live policy) on a hindsight-augmented prompt that embeds a
*successful sibling rollout* from the same prompt group, then distills the gap between the teacher's
and the (unconditioned) student's per-token log-probs over the sample's OWN response back into the
student (reverse KL; see ``compute_sdpo_loss`` in ``ppo_utils``).

This module builds the teacher inputs and is deliberately free of any task coupling: it operates only
on (uids, per-sample reward, decoded response texts, prompt chat-messages, tokenizer, ``SDPOConfig``).
SQL, tau-bench, SWE-agent, etc. differ only by the templates / threshold / feedback flags in the
config — the demonstration is the model's OWN successful rollout, so no gold label leaks.

Ported in spirit from the verl SDPO repo (``_collect_solutions_by_uid`` / ``_get_solution`` /
``_maybe_build_self_distillation_batch``), but using SkyRL's left-padded sequence convention so the
teacher's response tokens align position-for-position with the student's (and reuse the same
``loss_mask`` / ``num_actions``). See ``convert_prompts_responses_to_batch_tensors`` for that layout.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import torch
from loguru import logger

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def remove_thinking_trace(text: str) -> str:
    """Drop ``<think>...</think>`` spans (and trailing whitespace) from a demonstration."""
    return _THINK_RE.sub("", text)


def collect_success_by_uid(
    seq_rewards: Sequence[float],
    uids: Sequence[Any],
    success_reward_threshold: float,
) -> Dict[Any, List[int]]:
    """Group sample indices by prompt uid, keeping only those whose sequence reward clears the bar."""
    success_by_uid: Dict[Any, List[int]] = defaultdict(list)
    for idx, (uid, reward) in enumerate(zip(uids, seq_rewards)):
        if reward >= success_reward_threshold:
            success_by_uid[uid].append(idx)
    return success_by_uid


def get_demonstration(
    idx: int,
    success_by_uid: Dict[Any, List[int]],
    uids: Sequence[Any],
    response_texts: Sequence[str],
    dont_reprompt_on_self_success: bool,
    remove_thinking_from_demonstration: bool,
) -> Optional[str]:
    """Pick a successful sibling's response text as the demonstration for sample ``idx`` (or None)."""
    candidates = success_by_uid.get(uids[idx], [])
    if dont_reprompt_on_self_success:
        candidates = [j for j in candidates if j != idx]
    if not candidates:
        return None
    # First successful sibling — effectively a random correct rollout within the group.
    demonstration = response_texts[candidates[0]]
    if remove_thinking_from_demonstration:
        demonstration = remove_thinking_trace(demonstration)
    return demonstration


def build_hindsight_prompt_text(
    prompt_text: str,
    demonstration: Optional[str],
    feedback: Optional[str],
    cfg,
) -> str:
    """Render the hindsight user-turn text: original prompt + optional demonstration + optional feedback."""
    if demonstration is None and feedback is None:
        return prompt_text
    solution_section = (
        cfg.solution_template.format(successful_previous_attempt=demonstration) if demonstration is not None else ""
    )
    feedback_section = cfg.feedback_template.format(feedback_raw=feedback) if feedback is not None else ""
    return cfg.reprompt_template.format(prompt=prompt_text, solution=solution_section, feedback=feedback_section)


def _left_pad_concat(
    hindsight_prompt_ids: List[List[int]],
    response_ids_list: List[List[int]],
    pad_token_id: int,
) -> tuple:
    """Build left-padded teacher sequences ``[PAD..., hindsight_prompt, response]`` + attention masks.

    Mirrors verl: the response tokens are appended directly after the hindsight prompt (which ends
    with the assistant generation-prompt), so the response is conditioned exactly as it was generated
    — just with the demonstration injected upstream. Each row is left-padded to the batch max
    ``len(hindsight_i) + len(response_i)``; the response is the trailing real tokens, right-aligned in
    the ``[-num_actions-1:-1]`` slice to match the (shared) loss_mask. No clamping (hindsight >= 1, and
    ``len(response_i) <= num_actions``, so each row is ``>= num_actions + 1`` long)."""
    totals = [len(p) + len(r) for p, r in zip(hindsight_prompt_ids, response_ids_list)]
    max_total = max(totals)
    sequences, attention_masks = [], []
    for p, r, total in zip(hindsight_prompt_ids, response_ids_list, totals):
        pad_len = max_total - total
        sequences.append([pad_token_id] * pad_len + list(p) + list(r))
        attention_masks.append([0] * pad_len + [1] * total)
    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
    )


def build_self_distillation_tensors(
    tokenizer,
    prompt_messages: List[List[Dict[str, str]]],
    response_ids: List[List[int]],
    seq_rewards: Sequence[float],
    uids: Sequence[Any],
    cfg,
    feedback: Optional[Sequence[Optional[str]]] = None,
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the per-sample teacher inputs for SDPO.

    Args:
        tokenizer: model tokenizer (must support ``apply_chat_template`` and ``decode``).
        prompt_messages: per-sample chat messages (the same prompts fed to the generator). The final
            message is treated as the task/user turn whose content is augmented with hindsight.
        response_ids: per-sample response token ids (the SAME tokens used to build ``sequences``).
        seq_rewards: per-sample scalar sequence reward (e.g. ``rewards_tensor.sum(-1)``).
        uids: per-sample prompt-group id (the GRPO grouping key).
        cfg: ``SDPOConfig``.
        feedback: optional per-sample environment-feedback strings (None disables per sample). Left
            unused by the SQL minimal port; the seam for tau-bench user-sim / tool feedback.
        apply_chat_template_kwargs: extra kwargs forwarded to ``tokenizer.apply_chat_template``
            (e.g. ``{"enable_thinking": True}``).

    Returns:
        dict with CPU tensors ``teacher_sequences`` (B, S'), ``teacher_attention_mask`` (B, S'),
        ``self_distillation_mask`` (B, 1) float, and a ``metrics`` dict.
    """
    batch_size = len(response_ids)
    feedback = list(feedback) if feedback is not None else [None] * batch_size
    chat_kwargs = dict(apply_chat_template_kwargs or {})
    pad_token_id = tokenizer.pad_token_id

    response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids]
    success_by_uid = collect_success_by_uid(seq_rewards, uids, cfg.success_reward_threshold)

    hindsight_prompt_ids: List[List[int]] = []
    distillation_flags: List[float] = []
    num_with_demo = 0
    num_with_feedback_used = 0
    for i in range(batch_size):
        demonstration = get_demonstration(
            i,
            success_by_uid,
            uids,
            response_texts,
            cfg.dont_reprompt_on_self_success,
            cfg.remove_thinking_from_demonstration,
        )
        has_demo = demonstration is not None

        raw_feedback = feedback[i] if cfg.include_environment_feedback else None
        if raw_feedback is not None and not (isinstance(raw_feedback, str) and raw_feedback.strip()):
            raw_feedback = None
        # Optionally only use feedback when there is no demonstration.
        use_feedback = raw_feedback is not None and (
            not cfg.environment_feedback_only_without_solution or not has_demo
        )
        feedback_text = raw_feedback if use_feedback else None

        messages = prompt_messages[i]
        prefix = list(messages[:-1])
        prompt_text = messages[-1]["content"]
        hindsight_text = build_hindsight_prompt_text(prompt_text, demonstration, feedback_text, cfg)
        teacher_messages = prefix + [{"role": "user", "content": hindsight_text}]

        # Render to text then tokenize explicitly: apply_chat_template(tokenize=True) can return a
        # BatchEncoding (not List[int]) depending on the transformers version, which would corrupt the
        # token list. The two-step is version-robust and yields a flat List[int]. add_special_tokens=
        # False because the chat template already emits the special/format tokens.
        text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > cfg.max_reprompt_len:
            # Left-truncate (drop oldest tokens) so the generation-prompt suffix + final instruction survive.
            logger.warning(
                f"[sdpo] hindsight prompt for sample {i} ({len(ids)} tok) exceeds max_reprompt_len="
                f"{cfg.max_reprompt_len}; left-truncating."
            )
            ids = ids[-cfg.max_reprompt_len :]

        hindsight_prompt_ids.append(ids)
        distillation_flags.append(1.0 if (has_demo or use_feedback) else 0.0)
        num_with_demo += int(has_demo)
        num_with_feedback_used += int(use_feedback)

    teacher_sequences, teacher_attention_mask = _left_pad_concat(
        hindsight_prompt_ids, response_ids, pad_token_id
    )
    self_distillation_mask = torch.tensor(distillation_flags, dtype=torch.float32).unsqueeze(1)

    num_success_groups = sum(1 for v in success_by_uid.values() if len(v) > 0)
    num_groups = len(set(uids))
    metrics = {
        "sdpo/success_group_fraction": (num_success_groups / num_groups) if num_groups else 0.0,
        "sdpo/success_sample_fraction": num_with_demo / batch_size if batch_size else 0.0,
        "sdpo/feedback_used_fraction": num_with_feedback_used / batch_size if batch_size else 0.0,
        "sdpo/reprompt_sample_fraction": float(self_distillation_mask.mean().item()),
    }
    return {
        "teacher_sequences": teacher_sequences,
        "teacher_attention_mask": teacher_attention_mask,
        "self_distillation_mask": self_distillation_mask,
        "metrics": metrics,
    }
