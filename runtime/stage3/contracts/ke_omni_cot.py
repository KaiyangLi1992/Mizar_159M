"""Canonical Ke-Omni-R CoT prompt and strict completion contract.

The prompt builder intentionally mirrors Ke-Omni-R's ``_handle_avqa``
question template. Training, teacher inference, student rollout, and benchmark
evaluation must import this module rather than reimplementing the strings.
"""

from __future__ import annotations

import hashlib
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Sequence


SCHEMA_VERSION = "ke_omni_r_cot_prompt_contract_v1"
UPSTREAM_COMMIT = "915f44835e1e4d3307da1a990f787f15f6bd3e24"
UPSTREAM_DATASET_SOURCE_SHA256 = (
    "4a56896393011326073061af0b414a51c920e1b6201986b3ebe714ad079b6826"
)
THINK_MAX_WORDS = 50
COT_SUFFIX = (
    "Output the thinking process (less than 50 words) in <think> </think> "
    "and final answer in <answer> </answer>."
)

_COMPLETION_RE = re.compile(
    r"\A<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\Z",
    flags=re.DOTALL,
)
_TAGS = ("<think>", "</think>", "<answer>", "</answer>")
_CHOICE_LABEL_RE = re.compile(
    r"^\s*(?:\((?P<paren>[A-Za-z])\)|(?P<plain>[A-Za-z])[\)\].])\s*"
)


@dataclass(frozen=True)
class ParsedCompletion:
    thinking: str
    answer: str
    thinking_words: int


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def canonicalize_choices(choices: Sequence[str]) -> list[str]:
    """Remove an ordered label set and trailing empties without reordering.

    A prefix is a source label only when *every* choice carries the expected
    ordered A/B/C/... label.  This preserves semantic strings such as musical
    chord names (``D:maj(2)/2``), which must remain byte-identical to the
    upstream Ke list representation.
    """

    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
        raise TypeError("choices must be a sequence of strings")
    canonical = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, str):
            raise TypeError(f"choices[{index}] must be a string")
        canonical.append(choice.strip())
    while canonical and not canonical[-1]:
        canonical.pop()
    if len(canonical) < 2:
        raise ValueError("choices must contain at least two non-empty options")
    if any(not choice for choice in canonical):
        raise ValueError("internal empty choices are forbidden")
    matches = [_CHOICE_LABEL_RE.match(choice) for choice in canonical]
    labels = [
        ((match.group("paren") or match.group("plain")).lower() if match else None)
        for match in matches
    ]
    expected = [chr(ord("a") + index) for index in range(len(canonical))]
    if labels == expected:
        canonical = [
            choice[match.end() :].strip()
            for choice, match in zip(canonical, matches)
        ]
        if any(not choice for choice in canonical):
            raise ValueError("labeled choices must contain non-empty option text")
    return canonical


def labeled_choice_index(labeled_answer: str, choices: Sequence[str]) -> int:
    """Resolve and verify a source answer carrying an A/B/C/... prefix.

    Unlike :func:`answer_choice_index`, this intentionally permits duplicate
    choice text because the explicit source label identifies the gold slot.
    """

    labeled_answer = _require_text(labeled_answer, field="labeled_answer")
    match = _CHOICE_LABEL_RE.match(labeled_answer)
    if match is None:
        raise ValueError("labeled answer has no supported option prefix")
    label = (match.group("paren") or match.group("plain")).lower()
    index = ord(label) - ord("a")
    canonical = canonicalize_choices(choices)
    if not 0 <= index < len(canonical):
        raise ValueError("labeled answer index is outside the choice list")
    answer_text = labeled_answer[match.end() :].strip()
    if _normalize_choice_text(answer_text) != _normalize_choice_text(canonical[index]):
        raise ValueError("labeled answer text does not match its indexed choice")
    return index


def build_prompt(question: str, choices: Sequence[str]) -> str:
    """Build the official Ke-Omni-R think prompt for an ordered MCQ.

    This reproduces the upstream operations: lowercase ``video`` is replaced
    by ``audio`` and the ordered choices use Python's list representation.
    """

    question = _require_text(question, field="question")
    ordered_choices = canonicalize_choices(choices)

    question_text = question.replace("video", "audio")
    choice_text = f"Please choose the answer from the following options: {ordered_choices}."
    return f"{question_text} {choice_text} {COT_SUFFIX}"


def prompt_sha256(question: str, choices: Sequence[str]) -> str:
    return hashlib.sha256(build_prompt(question, choices).encode("utf-8")).hexdigest()


def parse_completion(text: str) -> ParsedCompletion:
    """Parse a completion with no direct-answer or malformed-tag fallback."""

    text = _require_text(text, field="completion")
    match = _COMPLETION_RE.fullmatch(text)
    if match is None:
        raise ValueError("completion does not match the strict think/answer format")

    thinking = match.group("think").strip()
    answer = match.group("answer").strip()
    if not thinking or not answer:
        raise ValueError("thinking and answer must both be non-empty")
    if any(tag in thinking or tag in answer for tag in _TAGS):
        raise ValueError("nested or repeated think/answer tags are forbidden")

    thinking_words = len(thinking.split())
    if thinking_words >= THINK_MAX_WORDS:
        raise ValueError(
            f"thinking must contain fewer than {THINK_MAX_WORDS} words; "
            f"got {thinking_words}"
        )
    return ParsedCompletion(
        thinking=thinking,
        answer=answer,
        thinking_words=thinking_words,
    )


def _normalize_choice_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def answer_choice_indices(answer: str, choices: Sequence[str]) -> list[int]:
    """Return every complete option text matched by ``answer``."""

    answer = _require_text(answer, field="answer")
    choices = canonicalize_choices(choices)
    # Ke-Omni-R's public reward first compares the extracted answer and gold
    # option by exact string equality.  Preserve that behavior before the
    # benchmark-tolerant fallback: punctuation folding would otherwise make
    # musical accidentals such as ``F# major`` collide with ``F major``.
    exact_answer = answer.strip()
    exact_matches = [
        index for index, choice in enumerate(choices) if exact_answer == choice.strip()
    ]
    if exact_matches:
        return exact_matches

    normalized_answer = _normalize_choice_text(answer)
    return [
        index
        for index, choice in enumerate(choices)
        if normalized_answer == _normalize_choice_text(
            _require_text(choice, field=f"choices[{index}]")
        )
    ]


def answer_choice_index(answer: str, choices: Sequence[str]) -> int:
    """Return the unique exact normalized option matched by ``answer``."""

    matches = answer_choice_indices(answer, choices)
    if len(matches) != 1:
        raise ValueError(
            "answer must match exactly one complete option after normalization"
        )
    return matches[0]


def completion_rewards(
    completion: str,
    choices: Sequence[str],
    gold_choice_index: int,
) -> tuple[float, float]:
    """Return Ke-style ``(accuracy, format)`` rewards under the strict parser.

    The public Ke reward extracts ``<answer>`` and compares it to the gold
    option using case-sensitive string equality.  The tolerant
    :func:`answer_choice_index` fallback is reserved for benchmark reporting
    and must not change the GRPO reward surface.
    """

    try:
        parsed = parse_completion(completion)
    except (TypeError, ValueError):
        return 0.0, 0.0

    format_reward = 1.0
    try:
        canonical = canonicalize_choices(choices)
    except (TypeError, ValueError):
        return 0.0, format_reward
    if type(gold_choice_index) is not int or not 0 <= gold_choice_index < len(canonical):
        return 0.0, format_reward
    return float(parsed.answer == canonical[gold_choice_index]), format_reward
