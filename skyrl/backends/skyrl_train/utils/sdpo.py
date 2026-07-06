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


def build_ground_truth_hindsight_text(prompt_text: str, ground_truth: str, cfg) -> str:
    """OPSD-style teacher user turn: original prompt + the gold framed as a *reference solution* +
    an anti-copy transition ("understand why it's correct, do NOT copy, now derive it yourself").

    This mirrors OPSD's non-``reason_first`` teacher prompt (lasgroup OPSD ``data_collator.py``): the
    point is to push the teacher's next-token distribution OFF literal answer-copying and onto a genuine
    independent-reasoning trajectory — which is what the answer-blind student can actually approximate.
    Used only when ``cfg.use_ground_truth_demonstration``; the sibling-demonstration path is unchanged.
    """
    reference = cfg.ground_truth_reference_template.format(ground_truth=ground_truth)
    return prompt_text + reference + cfg.ground_truth_transition_prompt


def build_sdpo_feedback(
    seq_rewards: Sequence[float],
    completions: Sequence[str],
    cfg,
) -> List[Optional[str]]:
    """Per-sample environment-feedback strings, keyed on the env reward (None where no feedback
    applies). Task-agnostic mechanism; the category wording lives in ``SDPOConfig``.

    Reward semantics (SQL three-valued {-1 bad format, 0 valid-but-wrong, +1 result-match}):
      * reward >= success_reward_threshold -> None (a success; it becomes a *demonstration* for
        its siblings, so it needs no failure feedback).
      * reward < 0 (format invalid) -> split on whether the response committed a <solution>:
        ``count("<solution>") == 0`` means it never committed (turn-exhaustion); ``>= 1`` means it
        committed but the format was still rejected (malformed). The count is over the response
        text only (the prompt's one-shot example is upstream and not included here).
      * otherwise (valid format, wrong result) -> wrong-result feedback.
    """
    out: List[Optional[str]] = []
    for reward, completion in zip(seq_rewards, completions):
        if reward >= cfg.success_reward_threshold:
            out.append(None)
        elif reward < 0:
            if completion.count("<solution>") == 0:
                out.append(cfg.feedback_no_commit)
            else:
                out.append(cfg.feedback_malformed)
        else:
            out.append(cfg.feedback_wrong_result)
    return out


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
    ground_truth: Optional[Sequence[Any]] = None,
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
        ground_truth: optional per-sample gold answers. When ``cfg.use_ground_truth_demonstration`` is
            set, the gold is framed as a *reference solution* + anti-copy transition (via
            ``cfg.ground_truth_reference_template`` + ``ground_truth_transition_prompt``) and REPLACES the
            sibling-demo / feedback hindsight prompt for the targeted samples (OPSD non-reason_first;
            label leak; works with n_samples=1). Ignored when the flag is off or the entry is None.
        apply_chat_template_kwargs: extra kwargs forwarded to ``tokenizer.apply_chat_template``
            (e.g. ``{"enable_thinking": True}``).

    Returns:
        dict with CPU tensors ``teacher_sequences`` (B, S'), ``teacher_attention_mask`` (B, S'),
        ``self_distillation_mask`` (B, 1) float, and a ``metrics`` dict.
    """
    batch_size = len(response_ids)
    feedback = list(feedback) if feedback is not None else [None] * batch_size
    ground_truth = list(ground_truth) if ground_truth is not None else [None] * batch_size
    chat_kwargs = dict(apply_chat_template_kwargs or {})
    pad_token_id = tokenizer.pad_token_id

    response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids]
    success_by_uid = collect_success_by_uid(seq_rewards, uids, cfg.success_reward_threshold)

    hindsight_prompt_ids: List[List[int]] = []
    distillation_flags: List[float] = []
    num_with_demo = 0
    num_with_feedback_used = 0
    num_with_ground_truth = 0
    use_gt = getattr(cfg, "use_ground_truth_demonstration", False)
    for i in range(batch_size):
        sibling_demo = get_demonstration(
            i,
            success_by_uid,
            uids,
            response_texts,
            cfg.dont_reprompt_on_self_success,
            cfg.remove_thinking_from_demonstration,
        )
        # OPSD-style gold conditioning: gate on use_ground_truth_demonstration (optionally only on failures).
        gold_i = ground_truth[i]
        use_gold = (
            use_gt
            and gold_i is not None
            and str(gold_i).strip() != ""
            and (not cfg.ground_truth_only_on_failure or seq_rewards[i] < cfg.success_reward_threshold)
        )
        has_sibling_demo = sibling_demo is not None

        messages = prompt_messages[i]
        prefix = list(messages[:-1])
        prompt_text = messages[-1]["content"]

        if use_gold:
            # Gold framed as a reference + anti-copy "derive it yourself" transition (OPSD non-reason_first).
            # Replaces the sibling-demo / feedback path entirely for this sample.
            hindsight_text = build_ground_truth_hindsight_text(prompt_text, str(gold_i).strip(), cfg)
            has_demo = True
            use_feedback = False
            num_with_ground_truth += 1
        else:
            raw_feedback = feedback[i] if cfg.include_environment_feedback else None
            if raw_feedback is not None and not (isinstance(raw_feedback, str) and raw_feedback.strip()):
                raw_feedback = None
            has_demo = sibling_demo is not None
            # Optionally only use feedback when there is no demonstration.
            use_feedback = raw_feedback is not None and (
                not cfg.environment_feedback_only_without_solution or not has_demo
            )
            feedback_text = raw_feedback if use_feedback else None
            hindsight_text = build_hindsight_prompt_text(prompt_text, sibling_demo, feedback_text, cfg)

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
        num_with_demo += int(has_sibling_demo)
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
        "sdpo/ground_truth_used_fraction": num_with_ground_truth / batch_size if batch_size else 0.0,
        "sdpo/reprompt_sample_fraction": float(self_distillation_mask.mean().item()),
    }
    return {
        "teacher_sequences": teacher_sequences,
        "teacher_attention_mask": teacher_attention_mask,
        "self_distillation_mask": self_distillation_mask,
        "metrics": metrics,
    }


def build_hero_perturn_tensors(
    tokenizer,
    prompt_messages: List[List[Dict[str, str]]],
    response_ids: List[List[int]],
    loss_masks: Sequence[Sequence[float]],
    hints: Sequence[Optional[dict]],
    seq_rewards: Sequence[float],
    cfg,
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build HERO faithful **per-turn** teacher inputs (Axis-B): one teacher row per problematic turn,
    each = ``H̃_t`` (history + Fig-3 feedback block) followed by that turn's exact ``y_t`` ids.

    Unlike :func:`build_self_distillation_tensors` (one row/sample, whole response), this emits
    ``Σ_i |𝓘(τ_i)|`` rows and an alignment map so the loss can gather each teacher row against the
    matching ``y_t`` slice of its sample's student forward. All the trajectory logic is delegated to the
    unit-tested pure module ``reverie.offline.hero_perturn``; this wrapper only packs tensors.

    Requires ``reverie`` on ``PYTHONPATH`` (box: ``PYTHONPATH=~/reverie``) and per-sample ``hints``
    (from the in-loop reflector, Chunk 3). Returns ``teacher_sequences=None`` when no turn qualifies.
    """
    from reverie.offline.chat_parse import parse_chat_trajectory
    from reverie.offline.hero_perturn import (
        assemble_teacher_rows,
        assistant_token_spans,
        build_teacher_specs,
    )

    threshold = cfg.success_reward_threshold
    scrub = getattr(cfg, "hero_scrub_args", False)
    all_specs: List[list] = []
    n_turn_mismatch = 0
    for i in range(len(response_ids)):
        # decode WITH special tokens so chat_parse sees the <|im_start|>/<|im_end|> turn markers.
        text = tokenizer.decode(response_ids[i], skip_special_tokens=False)
        turns = parse_chat_trajectory(text)
        spans = assistant_token_spans(loss_masks[i])
        if len(spans) != sum(1 for t in turns if t.role == "assistant"):
            n_turn_mismatch += 1  # loss-mask runs vs parsed assistant turns disagree (guarded downstream)
        reward = float(seq_rewards[i])
        all_specs.append(
            build_teacher_specs(
                turns,
                hints[i] or {},
                spans,
                prompt_messages=prompt_messages[i],
                trajectory_success=(reward >= threshold),
                final_reward=reward,
                scrub_args=scrub,
            )
        )

    prefix_ids_list, y_ids_list, row_meta, stats = assemble_teacher_rows(
        tokenizer, all_specs, response_ids,
        max_reprompt_len=cfg.max_reprompt_len,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    metrics = {
        "hero/n_teacher_rows": stats["n_teacher_rows"],
        "hero/n_prefix_truncated": stats["n_prefix_truncated"],
        "hero/n_turn_span_mismatch": n_turn_mismatch,
        "hero/frac_samples_with_turns": (
            sum(1 for s in all_specs if s) / len(all_specs) if all_specs else 0.0
        ),
    }
    if not row_meta:
        return {"teacher_sequences": None, "metrics": metrics}

    teacher_sequences, teacher_attention_mask = _left_pad_concat(
        prefix_ids_list, y_ids_list, tokenizer.pad_token_id
    )
    return {
        "teacher_sequences": teacher_sequences,               # (M, S') left-padded [PAD, H̃_t, y_t]
        "teacher_attention_mask": teacher_attention_mask,     # (M, S')
        "hero_sample_idx": torch.tensor([m["sample_idx"] for m in row_meta], dtype=torch.long),  # (M,)
        "hero_y_start": torch.tensor([m["y_start"] for m in row_meta], dtype=torch.long),        # (M,)
        "hero_y_end": torch.tensor([m["y_end"] for m in row_meta], dtype=torch.long),            # (M,)
        "hero_y_len": torch.tensor([m["y_len"] for m in row_meta], dtype=torch.long),            # (M,) scored tail
        "metrics": metrics,
    }
