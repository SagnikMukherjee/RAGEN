#!/usr/bin/env python3
"""
Repurpose ProAct Sokoban SFT CoTs into think-act-expect targets.

This script rewrites the assistant message in each row of a Sokoban SFT parquet
using Azure OpenAI, an OpenAI-compatible model router, or a local vLLM model.
It preserves the row id, board, answer, and prompt messages, and changes only
the assistant target to:

    <think>...</think>
    <act>...</act>
    <expect>...</expect>
    <answer>...</answer>

The <expect> block should summarize the consequence the original reasoning
implied for the chosen action.
"""

import argparse
import concurrent.futures
import copy
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_ENDPOINT = "https://sc-sm-m9rn3enk-eastus2.cognitiveservices.azure.com/"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_DEPLOYMENT = "gpt-4.1"
DEFAULT_MODEL_ROUTER_MODEL = "gpt-4o-mini"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-32B"
DEFAULT_INPUT_DIR = "data/proact_sokoban_sft"
DEFAULT_OUTPUT_DIR = "data/proact_sokoban_sft_think_act_expect"
REQUIRED_TAGS = ("think", "act", "expect", "answer")
ACTIONS = {"up": "Up", "down": "Down", "left": "Left", "right": "Right"}


SYSTEM_PROMPT = """You are a Sokoban-playing assistant.

Your job:
- Produce the final response for the current Sokoban state.
- Keep the reasoning natural-language, concrete, and concise.
- Output only the required tags, with no JSON, markdown, bullets outside tags, or extra text.
- The chosen action must exactly match the provided CHOSEN ACTION.
- Keep both <act> and <answer>; they must contain the same action.
- Write as the Sokoban-playing agent, not as a dataset editor.
- In <think>, discuss only the Sokoban board, player, boxes, targets, risks, and action choice.
- Never mention notes, private reference material, rewriting, the original response, the original answer, the prompt, the user, tags, formatting, or dataset construction.
- In <expect>, paraphrase only the expectation already stated in the private strategy notes for the chosen action.
- Treat the private strategy notes as the source of truth, even if their coordinates or details differ from the board.
- Do not use the board or current state to correct, normalize, or add coordinates.
- Preserve the coordinate frame and level of detail used in the private strategy notes.
- Include concrete state consequences when explicitly stated, such as moved boxes, player position, target progress, avoided risk, or future access.
- Keep <expect> concise: one to three natural-language sentences.

Required format:
<think>
Natural-language reasoning about the current Sokoban state and why the chosen action is sensible.
</think>
<act>
CHOSEN ACTION
</act>
<expect>
Pushing the chosen direction should move the relevant box onto or toward its target, leave the other boxes in place, and put the player in position to continue the planned route.
</expect>
<answer>
CHOSEN ACTION
</answer>"""


EXPECT_SYSTEM_PROMPT = """You paraphrase the expectation stated in Sokoban reasoning.

Your job:
- Write a concise natural-language paraphrase of what the agent expected would happen after the chosen action.
- Use the private strategy notes as the source of truth.
- Use the current state only for broad context, not to correct or normalize the notes.
- Do not invent a different plan, choose a different action, use simulator ground truth, or repair mistakes.
- Include coordinates only when they are stated in the notes, preserving the notes' coordinate frame and wording style.
- Include strategic consequences only when present in the notes, such as target progress, deadlock avoidance, future access, or player positioning.
- Do not include <think>, <act>, <answer>, JSON, markdown, or bullets.
- You may include <expect>...</expect>, but plain text is preferred.
- Discuss only the expectation stated in the reasoning for the chosen action.
- Do not use the phrase "the chosen action" in the output; say "Moving Up", "Moving Left", or "This move" instead.
- Never mention notes, private reference material, rewriting, prompts, tags, formatting, users, or dataset construction.
- Keep the output to one to three sentences.

Examples:

If the reasoning expects a direct target push:
Pushing Up should move the box at (3, 1) onto the target at (2, 1), with the player ending at (3, 1). This secures one target while keeping access to the remaining box.

If the reasoning expects a setup move:
Moving Right should reposition the player beside the next box without changing the boxes yet. This should set up a later push while avoiding the risky corner.

If the reasoning is high-level:
The action should make immediate progress toward the nearest target and avoid pushing the box into a deadlocked wall pocket.

If the reasoning uses row/col coordinates:
Moving Right should let the worker step to row4 col5, then row3 col5, so it can push left and slide the box from col4 onto the target at col3. This is expected to complete the placement safely without pushing the box toward the wall.
"""


def build_user_prompt(row: pd.Series) -> str:
    messages = row["messages"]
    system_content, user_content, assistant_content = split_messages(messages)
    strategy_notes = extract_tag(assistant_content, "think") or assistant_content
    user_content = strip_old_format_instruction(user_content)
    return f"""Produce the final Sokoban assistant response for this state.

Use the private strategy notes only to preserve the intended Sokoban plan. Do not mention the notes.
Write <think> as if you are directly solving the Sokoban state.

CHOSEN ACTION:
{row["answer"]}

BOARD COLUMN:
{row["board"]}

GAME INSTRUCTIONS:
{system_content}

CURRENT STATE:
{user_content}

PRIVATE STRATEGY NOTES:
{strategy_notes}

Return only:
<think>...</think>
<act>{row["answer"]}</act>
<expect>...</expect>
<answer>{row["answer"]}</answer>
"""


def build_expect_user_prompt(row: pd.Series) -> str:
    messages = row["messages"]
    system_content, user_content, assistant_content = split_messages(messages)
    strategy_notes = extract_tag(assistant_content, "think") or assistant_content
    user_content = strip_old_format_instruction(user_content)
    return f"""Write only the <expect> content for this Sokoban action.

CHOSEN ACTION:
{row["answer"]}

BOARD COLUMN:
{row["board"]}

GAME INSTRUCTIONS:
{system_content}

CURRENT STATE:
{user_content}

PRIVATE STRATEGY NOTES:
{strategy_notes}

Paraphrase the expectation the agent stated when choosing this action.

Rules:
- Use only the private strategy notes as the source for the expectation.
- Do not use the current state or board to correct, normalize, or add coordinates.
- Preserve the coordinate frame used in the private strategy notes. If the notes say row3 col4, do not rewrite it as (2, 3).
- Do not add consequences that are not stated or directly paraphrased from the private strategy notes.
- Do not compute a new answer or replace the chosen action.
- Include strategic beliefs if they are present: target progress, avoided deadlock, future route/access, or player positioning.
- Keep it concise: one to three natural-language sentences.
- Do not write "the chosen action"; refer to the move directly, such as "Moving Up" or "This move".
- Do not mention private notes, extraction, prompts, tags, formatting, or dataset construction.

Good examples:
Pushing Up should move the box at (3, 1) onto the target at (2, 1), with the player ending at (3, 1). This secures one target and leaves the route to the other box open.

Moving Right should reposition the player toward the next box without changing box locations yet. The expected benefit is improved access for the following push while avoiding a deadlock-prone side pocket.

Moving Right should let the worker step to row4 col5, then row3 col5, so it can push left and slide the box from col4 onto the target at col3. This is expected to complete the placement safely without pushing the box toward the wall.

Bad examples:
The notes say that I should extract an expectation.
Boxes: (6, 2)
Player: (5, 3)
The box at row3 col4 should move to (2, 2).
"""


def strip_old_format_instruction(text: str) -> str:
    """Remove stale target-format instructions from the original SFT prompt."""
    kept = []
    for line in str(text).splitlines():
        lowered = line.lower()
        if "always output:" in lowered:
            continue
        if "strictly follow this format" in lowered:
            continue
        if "max response length" in lowered:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def split_messages(messages: Any) -> Tuple[str, str, str]:
    """Return system, user, assistant content from a parquet messages cell."""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("messages must be a list with at least three chat messages")

    try:
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]
        assistant_content = messages[2]["content"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("messages must contain dicts with content fields") from exc

    return str(system_content), str(user_content), str(assistant_content)


def normalize_action(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return ACTIONS.get(cleaned.lower(), cleaned)


def extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def count_tag_pairs(text: str, tag: str) -> int:
    opens = len(re.findall(fr"<{tag}>", text, flags=re.IGNORECASE))
    closes = len(re.findall(fr"</{tag}>", text, flags=re.IGNORECASE))
    return min(opens, closes)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|xml|html)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def force_action_tags(text: str, answer: str) -> str:
    """Replace existing <act> and <answer> bodies with the canonical answer."""
    canonical = normalize_action(answer)
    for tag in ("act", "answer"):
        text = re.sub(
            fr"(<{tag}>\s*)(.*?)(\s*</{tag}>)",
            fr"\1{canonical}\3",
            text,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return text


def validate_response(text: str, answer: str) -> Tuple[bool, str]:
    if not isinstance(text, str) or not text.strip():
        return False, "empty response"

    text = strip_code_fences(text)
    for tag in REQUIRED_TAGS:
        if count_tag_pairs(text, tag) != 1:
            return False, f"expected exactly one <{tag}>...</{tag}> block"

    # Require no non-whitespace text outside the four expected tags in order.
    pattern = (
        r"^\s*<think>\s*.+?\s*</think>\s*"
        r"<act>\s*.+?\s*</act>\s*"
        r"<expect>\s*.+?\s*</expect>\s*"
        r"<answer>\s*.+?\s*</answer>\s*$"
    )
    if not re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE):
        return False, "response must contain only think, act, expect, answer blocks in order"

    expected_action = normalize_action(answer)
    act = normalize_action(extract_tag(text, "act") or "")
    final_answer = normalize_action(extract_tag(text, "answer") or "")
    if act != expected_action:
        return False, f"<act> mismatch: expected {expected_action!r}, got {act!r}"
    if final_answer != expected_action:
        return False, f"<answer> mismatch: expected {expected_action!r}, got {final_answer!r}"

    think = extract_tag(text, "think") or ""
    expect = extract_tag(text, "expect") or ""
    if len(think.split()) < 20:
        return False, "<think> is too short"

    meta_regex = (
        r"\b("
        r"dataset|"
        r"private\s+(?:strategy\s+)?notes|strategy\s+notes|"
        r"original\s+(?:answer|assistant|response|cot)|"
        r"assistant(?:'s)?\s+response|"
        r"user\s+(?:wants|asked|provided|prompt)|"
        r"prompt|format|tag|tags|"
        r"need\s+to\s+(?:make\s+sure|structure|include|output)|"
        r"required\s+format"
        r")\b"
    )
    for block_name, block_text in (("think", think), ("expect", expect)):
        if re.search(meta_regex, block_text, flags=re.IGNORECASE):
            return False, f"<{block_name}> contains meta rewriting/formatting language"

    if len(expect.split()) < 8:
        return False, "<expect> is too short"
    if len(expect.split()) > 120:
        return False, "<expect> is too long"
    if re.search(r"</?(?:think|act|answer)>", expect, flags=re.IGNORECASE):
        return False, "<expect> contains another response tag"

    return True, ""


def make_client(args: argparse.Namespace):
    from openai import AzureOpenAI

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing Azure OpenAI API key. Set {args.api_key_env}=... before running."
        )

    return AzureOpenAI(
        api_version=args.api_version,
        azure_endpoint=args.endpoint,
        api_key=api_key,
    )


def make_model_router_client(args: argparse.Namespace):
    from openai import OpenAI

    api_key = os.environ.get(args.model_router_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing model-router API key. Set {args.model_router_api_key_env}=... before running."
        )

    base_url = args.model_router_base_url or os.environ.get(args.model_router_base_url_env)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def build_generation_messages(row: pd.Series, args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.rewrite_mode == "append-expect":
        return [
            {"role": "system", "content": EXPECT_SYSTEM_PROMPT},
            {"role": "user", "content": build_expect_user_prompt(row)},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]


def call_azure(client: Any, args: argparse.Namespace, row: pd.Series) -> str:
    response = client.chat.completions.create(
        model=args.deployment,
        messages=build_generation_messages(row, args),
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content or ""


def call_model_router(client: Any, args: argparse.Namespace, row: pd.Series) -> str:
    response = client.chat.completions.create(
        model=args.model_router_model,
        messages=build_generation_messages(row, args),
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content or ""


def preflight_azure(client: Any, args: argparse.Namespace) -> None:
    """Make a tiny request before launching a long dataset rewrite."""
    response = client.chat.completions.create(
        model=args.deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with exactly: endpoint ok"},
        ],
        max_completion_tokens=32,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Azure OpenAI preflight returned an empty response")
    print(f"[preflight] Azure OpenAI response: {content}")


def preflight_model_router(client: Any, args: argparse.Namespace) -> None:
    """Make a tiny request before launching a long dataset rewrite."""
    response = client.chat.completions.create(
        model=args.model_router_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with exactly: endpoint ok"},
        ],
        max_completion_tokens=32,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("model-router preflight returned an empty response")
    print(f"[preflight] model-router response: {content}")


def make_vllm(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_tokens,
    )
    return llm, tokenizer, sampling_params


def build_chat_prompt(tokenizer: Any, row: pd.Series, args: argparse.Namespace) -> str:
    messages = build_generation_messages(row, args)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args_supports_disable_thinking(tokenizer):
        kwargs["enable_thinking"] = False
    try:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    if args.provider == "vllm" and args.rewrite_mode == "full" and args.assistant_prefix:
        prompt += args.assistant_prefix
    return prompt


def args_supports_disable_thinking(tokenizer: Any) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        return False
    return "enable_thinking" in chat_template


def finalize_candidate(
    raw: str,
    answer: str,
    args: argparse.Namespace,
    row: Optional[pd.Series] = None,
) -> Tuple[bool, str, str]:
    if args.rewrite_mode == "append-expect":
        if row is None:
            return False, "", "append-expect mode requires source row"
        candidate = build_appended_expect_response(row, raw, answer)
        ok, error = validate_response(candidate, answer)
        return ok, candidate, error

    candidate = strip_code_fences(raw)
    if args.provider == "vllm":
        candidate = repair_vllm_candidate(candidate, answer, args)
    if args.force_action_tags:
        candidate = force_action_tags(candidate, answer)
    ok, error = validate_response(candidate, answer)
    return ok, candidate, error


def build_appended_expect_response(row: pd.Series, raw_expectation: str, answer: str) -> str:
    _, _, original_assistant = split_messages(row["messages"])
    think = extract_tag(original_assistant, "think")
    if not think:
        answerless = re.sub(r"<answer>.*?</answer>", "", original_assistant, flags=re.DOTALL | re.IGNORECASE)
        think = answerless.strip()
    canonical = normalize_action(answer)
    expectation = clean_expectation_text(extract_expectation_text(raw_expectation), canonical)
    return (
        f"<think>\n{think.strip()}\n</think>\n"
        f"<act>{canonical}</act>\n"
        f"<expect>\n{expectation.strip()}\n</expect>\n"
        f"<answer>{canonical}</answer>"
    )


def extract_expectation_text(text: str) -> str:
    text = strip_code_fences(text)
    expect = extract_tag(text, "expect")
    if expect:
        return expect
    text = re.sub(r"</?(?:think|act|answer)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?expect>", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_expectation_text(text: str, action: str) -> str:
    """Lightly smooth common generator boilerplate in expectation text."""
    text = text.strip()
    text = re.sub(
        r"^\s*the\s+chosen\s+action\s+[\"']?([A-Za-z]+)[\"']?\s+is\s+legal\b",
        lambda m: f"Moving {normalize_action(m.group(1))} is legal",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*the\s+chosen\s+action\s+[\"']?([A-Za-z]+)[\"']?\s+is\s+illegal\b",
        lambda m: f"Moving {normalize_action(m.group(1))} is illegal",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bthe\s+chosen\s+action\s+[\"']?%s[\"']?\b" % re.escape(action),
        f"moving {action}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bthe\s+chosen\s+action\b",
        "this move",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*this move\b", "This move", text, flags=re.IGNORECASE)
    return text


def repair_vllm_candidate(text: str, answer: str, args: argparse.Namespace) -> str:
    """Repair common local-model formatting misses without changing semantics."""
    text = strip_code_fences(text)
    stripped = text.strip()
    canonical = normalize_action(answer)

    if args.assistant_prefix and stripped.lower().startswith(args.assistant_prefix.strip().lower()):
        stripped = re.sub(
            r"^\s*<think>\s*",
            "",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

    tag_count = sum(count_tag_pairs(stripped, tag) for tag in REQUIRED_TAGS)
    has_any_tag = bool(re.search(r"</?(?:think|act|expect|answer)>", stripped, flags=re.IGNORECASE))

    # Qwen often returns only the continuation after the <think> prefill. If the
    # continuation is plain Sokoban prose, wrap it into the requested structure.
    if not has_any_tag and stripped:
        return (
            f"<think>\n{stripped}\n</think>\n"
            f"<act>{canonical}</act>\n"
            f"<expect>\n{stripped}\n</expect>\n"
            f"<answer>{canonical}</answer>"
        )

    # If there is only an unclosed think block, close it and add the action
    # scaffold. This keeps the generated prose intact.
    if tag_count == 0 and re.search(r"<think>", stripped, flags=re.IGNORECASE):
        body = re.sub(r"^\s*<think>\s*", "", stripped, count=1, flags=re.IGNORECASE).strip()
        return (
            f"<think>\n{body}\n</think>\n"
            f"<act>{canonical}</act>\n"
            f"<expect>\n{body}\n</expect>\n"
            f"<answer>{canonical}</answer>"
        )

    return stripped


def update_target_format_instruction(text: str) -> str:
    """Replace the original two-tag target instruction with the repurposed format."""
    replacement = (
        "You have 1 actions left. Always output: "
        "<think> [Your thoughts] </think> "
        "<act> [your action] </act> "
        "<expect> [expected consequence of your action] </expect> "
        "<answer> [your answer] </answer> "
        "with no extra text. Strictly follow this format. "
        "In <expect>, briefly describe what you expect to happen after your action. "
        "Max response length: 400 words (tokens)."
    )
    pattern = (
        r"You have (?P<count>\d+) actions left\.\s*"
        r"Always output:.*?"
        r"Max response length:\s*400 words \(tokens\)\."
    )

    def repl(match: re.Match) -> str:
        return replacement.replace("1 actions", f"{match.group('count')} actions")

    updated = re.sub(pattern, repl, str(text), count=1, flags=re.DOTALL)
    return updated


def update_assistant_message(messages: Any, new_assistant: str) -> List[Dict[str, str]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    new_messages = copy.deepcopy(messages)
    if not isinstance(new_messages, list) or len(new_messages) < 3:
        raise ValueError("messages must be a list with at least three chat messages")
    new_messages[1]["content"] = update_target_format_instruction(new_messages[1]["content"])
    new_messages[2]["content"] = new_assistant
    return new_messages


def row_to_record(
    row: pd.Series,
    new_assistant: str,
    status: str,
    error: str = "",
    raw_generation: str = "",
) -> Dict[str, Any]:
    _, _, original_assistant = split_messages(row["messages"])
    record = row.to_dict()
    record["messages"] = update_assistant_message(row["messages"], new_assistant)
    record["repurpose_status"] = status
    record["original_assistant"] = original_assistant
    record["repurpose_error"] = error
    record["repurpose_raw_generation"] = raw_generation
    return record


def append_jsonl(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Ignoring malformed checkpoint line {line_no}: {path}", file=sys.stderr)
                continue
            records[str(record["id"])] = record
    return records


def process_one(
    row_index: int,
    row: pd.Series,
    client: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row_id = row["id"]
    original_answer = normalize_action(str(row["answer"]))
    last_error = ""

    for attempt in range(args.retries + 1):
        try:
            if args.provider == "model-router":
                raw = call_model_router(client, args, row)
            else:
                raw = call_azure(client, args, row)
            ok, candidate, error = finalize_candidate(raw, original_answer, args, row)
            if ok:
                return row_to_record(row, candidate, "ok")
            last_error = error
        except Exception as exc:  # noqa: BLE001 - preserve API/validation errors in checkpoint
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < args.retries:
            sleep_s = args.retry_base_delay * (2 ** attempt)
            sleep_s += min(0.5, 0.05 * (row_index % 10))
            time.sleep(sleep_s)

    _, _, original_assistant = split_messages(row["messages"])
    return row_to_record(row, original_assistant, "failed", last_error)


def process_vllm_rows(
    split: str,
    rows: pd.DataFrame,
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_lock: threading.Lock,
) -> Dict[str, Dict[str, Any]]:
    """Generate rows with vLLM in batches and retry only invalid generations."""
    new_records: Dict[str, Dict[str, Any]] = {}
    pending: List[Tuple[int, pd.Series, str]] = [
        (row_index, row, "")
        for row_index, row in rows.iterrows()
    ]
    total = len(pending)
    processed_count = 0

    for attempt in range(args.retries + 1):
        if not pending:
            break

        next_pending: List[Tuple[int, pd.Series, str]] = []
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts = [build_chat_prompt(tokenizer, row, args) for _, row, _ in batch]
            outputs = llm.generate(prompts, sampling_params)

            for (row_index, row, previous_error), output in zip(batch, outputs):
                row_id = str(row["id"])
                answer = normalize_action(str(row["answer"]))
                raw = output.outputs[0].text if output.outputs else ""
                ok, candidate, error = finalize_candidate(raw, answer, args, row)
                if ok:
                    record = row_to_record(row, candidate, "ok")
                    new_records[row_id] = record
                    append_jsonl(checkpoint_path, record, checkpoint_lock)
                    processed_count += 1
                    print_progress(split, processed_count, total, record)
                elif attempt < args.retries:
                    next_pending.append((row_index, row, error))
                else:
                    _, _, original_assistant = split_messages(row["messages"])
                    final_error = error or previous_error or "invalid vLLM generation"
                    record = row_to_record(
                        row,
                        original_assistant,
                        "failed",
                        final_error,
                        raw_generation=raw,
                    )
                    new_records[row_id] = record
                    append_jsonl(checkpoint_path, record, checkpoint_lock)
                    processed_count += 1
                    print_progress(split, processed_count, total, record)

        pending = next_pending
        if pending:
            print(
                f"[{split}] retrying {len(pending)} invalid vLLM generations "
                f"(attempt {attempt + 1}/{args.retries})",
                flush=True,
            )

    return new_records


def materialize_records(
    df: pd.DataFrame,
    completed: Dict[str, Dict[str, Any]],
    pending_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        row_id = str(row["id"])
        if row_id in pending_results:
            records.append(pending_results[row_id])
        elif row_id in completed:
            records.append(completed[row_id])
        else:
            _, _, original_assistant = split_messages(row["messages"])
            records.append(row_to_record(row, original_assistant, "skipped"))
    return pd.DataFrame(records)


def select_rows(
    df: pd.DataFrame,
    completed: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    selected = df
    if args.resume and not args.overwrite:
        ok_ids = {
            row_id
            for row_id, record in completed.items()
            if record.get("repurpose_status") == "ok"
        }
        selected = selected[~selected["id"].astype(str).isin(ok_ids)]
    if args.limit is not None:
        selected = selected.head(args.limit)
    return selected


def process_split(split: str, args: argparse.Namespace) -> None:
    input_path = Path(args.input_dir) / f"{split}.parquet"
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{split}.parquet"
    checkpoint_path = output_dir / f"{split}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input split: {input_path}")

    df = pd.read_parquet(input_path)
    validate_dataframe(df, input_path)
    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    rows = select_rows(df, completed, args)

    print(
        f"[{split}] input_rows={len(df)} completed={len(completed)} "
        f"to_process={len(rows)} output={output_path}"
    )

    if args.dry_run:
        if len(rows) == 0:
            print(f"[{split}] no rows selected for dry run")
            return
        row = rows.iloc[0]
        print("\n===== SYSTEM PROMPT =====\n")
        print(EXPECT_SYSTEM_PROMPT if args.rewrite_mode == "append-expect" else SYSTEM_PROMPT)
        print("\n===== USER PROMPT =====\n")
        print(build_expect_user_prompt(row) if args.rewrite_mode == "append-expect" else build_user_prompt(row))
        print("\n===== END DRY RUN =====")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Checkpoint exists: {checkpoint_path}. Use --resume or --overwrite."
        )
    if output_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --resume or --overwrite."
        )
    if args.overwrite:
        completed = {}
        checkpoint_path.unlink(missing_ok=True)

    checkpoint_lock = threading.Lock()
    new_records: Dict[str, Dict[str, Any]] = {}

    if args.provider == "vllm":
        print(
            f"[{split}] loading vLLM model={args.model} "
            f"tensor_parallel_size={args.tensor_parallel_size}"
        )
        llm, tokenizer, sampling_params = make_vllm(args)
        new_records = process_vllm_rows(
            split,
            rows,
            llm,
            tokenizer,
            sampling_params,
            args,
            checkpoint_path,
            checkpoint_lock,
        )
    elif args.provider == "model-router":
        client = make_model_router_client(args)
        if not args.no_preflight:
            model_name = args.model_router_model
            print(f"[{split}] running model-router preflight model={model_name}...")
            preflight_model_router(client, args)

        if args.workers == 1:
            iterator: Iterable[Tuple[int, pd.Series]] = rows.iterrows()
            for processed_count, (row_index, row) in enumerate(iterator, start=1):
                record = process_one(row_index, row, client, args)
                new_records[str(record["id"])] = record
                append_jsonl(checkpoint_path, record, checkpoint_lock)
                print_progress(split, processed_count, len(rows), record)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_index = {
                    executor.submit(process_one, row_index, row, client, args): row_index
                    for row_index, row in rows.iterrows()
                }
                for processed_count, future in enumerate(
                    concurrent.futures.as_completed(future_to_index),
                    start=1,
                ):
                    record = future.result()
                    new_records[str(record["id"])] = record
                    append_jsonl(checkpoint_path, record, checkpoint_lock)
                    print_progress(split, processed_count, len(rows), record)
    else:
        client = make_client(args)
        if not args.no_preflight:
            print(f"[{split}] running Azure OpenAI preflight...")
            preflight_azure(client, args)

        if args.workers == 1:
            iterator: Iterable[Tuple[int, pd.Series]] = rows.iterrows()
            for processed_count, (row_index, row) in enumerate(iterator, start=1):
                record = process_one(row_index, row, client, args)
                new_records[str(record["id"])] = record
                append_jsonl(checkpoint_path, record, checkpoint_lock)
                print_progress(split, processed_count, len(rows), record)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_index = {
                    executor.submit(process_one, row_index, row, client, args): row_index
                    for row_index, row in rows.iterrows()
                }
                for processed_count, future in enumerate(
                    concurrent.futures.as_completed(future_to_index),
                    start=1,
                ):
                    record = future.result()
                    new_records[str(record["id"])] = record
                    append_jsonl(checkpoint_path, record, checkpoint_lock)
                    print_progress(split, processed_count, len(rows), record)

    merged = materialize_records(df, completed, new_records)
    merged.to_parquet(output_path, index=False)
    status_counts = merged["repurpose_status"].value_counts(dropna=False).to_dict()
    print(f"[{split}] wrote {output_path}")
    print(f"[{split}] status_counts={status_counts}")


def print_progress(split: str, processed_count: int, total: int, record: Dict[str, Any]) -> None:
    status = record.get("repurpose_status", "?")
    row_id = record.get("id", "?")
    suffix = ""
    if status != "ok":
        suffix = f" error={record.get('repurpose_error', '')}"
    print(f"[{split}] {processed_count}/{total} id={row_id} status={status}{suffix}", flush=True)


def validate_dataframe(df: pd.DataFrame, path: Path) -> None:
    required = {"id", "messages", "board", "answer"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repurpose ProAct Sokoban SFT CoTs into think-act-expect targets."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--provider", choices=["azure", "model-router", "vllm"], default="azure")
    parser.add_argument(
        "--rewrite-mode",
        choices=["full", "append-expect"],
        default="full",
        help="full regenerates think/act/expect/answer; append-expect keeps original think/answer and generates only expect.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION)
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--api-key-env", default="AZURE_OPENAI_API_KEY")
    parser.add_argument(
        "--model-router-model",
        default=DEFAULT_MODEL_ROUTER_MODEL,
        help="Model name for --provider model-router, e.g. gpt-4o-mini.",
    )
    parser.add_argument(
        "--model-router-api-key-env",
        default="MODEL_ROUTER_API_KEY",
        help="Environment variable containing the model-router API key.",
    )
    parser.add_argument(
        "--model-router-base-url-env",
        default="MODEL_ROUTER_BASE_URL",
        help="Optional environment variable containing an OpenAI-compatible router base URL.",
    )
    parser.add_argument(
        "--model-router-base-url",
        default=None,
        help="Optional OpenAI-compatible router base URL. Overrides --model-router-base-url-env.",
    )
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL, help="Local/HF model path for --provider vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--assistant-prefix",
        default="<think>\n",
        help="Text appended after the chat template for vLLM and prepended before validation.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the tiny API connectivity test before processing rows.",
    )
    parser.add_argument(
        "--no-force-action-tags",
        dest="force_action_tags",
        action="store_false",
        help="Do not rewrite <act>/<answer> bodies to match the answer column before validation.",
    )
    parser.set_defaults(force_action_tags=True)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    for split in args.splits:
        process_split(split, args)


if __name__ == "__main__":
    main()
