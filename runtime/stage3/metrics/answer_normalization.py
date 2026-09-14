import re
import string


ARTICLES = {"a", "an", "the"}
MCQ_OPTIONS = ("a", "b", "c", "d")


def normalize_answer(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def extract_mcq_option(text, choices=None):
    text = "" if text is None else str(text)
    stripped = text.strip().lower()
    if stripped in MCQ_OPTIONS:
        return stripped

    patterns = [
        r"(?:^|[\s:\(\[])([abcd])\s*[\)\].:]",
        r"\b(?:answer|option|choice)\s*[:\-]?\s*\(?([abcd])\)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()

    if choices:
        normalized_text = normalize_answer(text)
        for idx, choice in enumerate(choices):
            if idx >= len(MCQ_OPTIONS):
                break
            normalized_choice = normalize_answer(choice)
            if normalized_choice and normalized_choice in normalized_text:
                return MCQ_OPTIONS[idx]

    return None


def extract_yes_no(text):
    text = "" if text is None else str(text)
    match = re.search(r"\b(yes|no)\b", text.lower())
    return match.group(1) if match else None


def normalize_prediction_for_choices(prediction, choices=None):
    option = extract_mcq_option(prediction, choices)
    if option is not None and choices:
        idx = MCQ_OPTIONS.index(option)
        if idx < len(choices):
            return str(choices[idx]).strip()

    if choices:
        normalized_prediction = normalize_answer(prediction)
        for choice in choices:
            normalized_choice = normalize_answer(choice)
            if normalized_choice and normalized_choice in normalized_prediction:
                return str(choice).strip()

    yes_no = extract_yes_no(prediction)
    if yes_no is not None:
        return yes_no

    return normalize_answer(prediction)
