
from metrics.answer_normalization import extract_mcq_option, normalize_answer


def is_correct_mcq(response, answer):
    response_option = extract_mcq_option(response)
    answer_option = extract_mcq_option(answer)
    if response_option is not None and answer_option is not None:
        return response_option == answer_option
    return normalize_answer(response) == normalize_answer(answer)


def evaluate_metric(preds, answers, metadata):
    corr = 0
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    # compute metrics
    for i in range(len(preds)):
        answer = answers[i]
        response = preds[i]
        correct = is_correct_mcq(response, answer)

        task = metadata[i]['task']
        difficulty = metadata[i]['difficulty']

        if correct:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            corr += 1

        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1

    # Parse, collect and return metrics
    scores = {t: {} for t in ['sound','music','speech','easy','hard','medium','total','main']}
    for task in task_metrics:
        scores[task]['score'] = (task_metrics[task][0]/task_metrics[task][1])*100 if task_metrics[task][1] != 0 else 0
    for diff in diff_metrics:
        scores[diff]['score'] = (diff_metrics[diff][0]/diff_metrics[diff][1])*100 if diff_metrics[diff][1] != 0 else 0
    scores["total"]['score'] = (corr/len(preds)) * 100
    scores["main"]["score"] = scores["total"]['score']
    return scores

def save_json(preds, answers, metadata):
    corr = 0
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    # compute metrics
    for i in range(len(preds)):
        answer = answers[i]
        response = preds[i]
        correct = is_correct_mcq(response, answer)

        task = metadata[i]['task']
        difficulty = metadata[i]['difficulty']

        if correct:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            corr += 1

        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1

    # Parse, collect and return metrics
    scores = {t: {} for t in ['sound','music','speech','easy','hard','medium','total','main']}
    for task in task_metrics:
        scores[task]['score'] = (task_metrics[task][0]/task_metrics[task][1])*100 if task_metrics[task][1] != 0 else 0
    for diff in diff_metrics:
        scores[diff]['score'] = (diff_metrics[diff][0]/diff_metrics[diff][1])*100 if diff_metrics[diff][1] != 0 else 0
    scores["total"]['score'] = (corr/len(preds)) * 100
    scores["main"]["score"] = scores["total"]['score']
    return scores
