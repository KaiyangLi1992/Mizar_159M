from collections import defaultdict

from metrics.answer_normalization import extract_mcq_option, normalize_answer


def token_f1(prediction, answer):
    pred_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0

    common = defaultdict(int)
    for token in answer_tokens:
        common[token] += 1

    overlap = 0
    for token in pred_tokens:
        if common[token] > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_metric(preds, answers, metadata=None):
    exact = 0
    f1_sum = 0.0
    mcq_total = 0
    mcq_correct = 0
    subtype = defaultdict(lambda: [0.0, 0])

    metadata = metadata or [{} for _ in answers]
    for pred, answer, meta in zip(preds, answers, metadata):
        pred_norm = normalize_answer(pred)
        answer_norm = normalize_answer(answer)
        if pred_norm == answer_norm:
            exact += 1

        f1 = token_f1(pred, answer)
        f1_sum += f1

        pred_option = extract_mcq_option(pred)
        answer_option = extract_mcq_option(answer)
        if answer_option is not None:
            mcq_total += 1
            if pred_option == answer_option:
                mcq_correct += 1

        key = meta.get("subtype") or meta.get("taskname") or "unknown"
        subtype[key][0] += f1
        subtype[key][1] += 1

    total = len(answers)
    scores = {
        "EM": {"score": (exact / total) * 100 if total else 0},
        "F1": {"score": (f1_sum / total) * 100 if total else 0},
        "MCQ_ACC": {"score": (mcq_correct / mcq_total) * 100 if mcq_total else 0},
        "main": {"score": (f1_sum / total) * 100 if total else 0},
    }

    for key, (score_sum, count) in subtype.items():
        scores[f"F1_{key}"] = {"score": (score_sum / count) * 100 if count else 0}

    return scores
