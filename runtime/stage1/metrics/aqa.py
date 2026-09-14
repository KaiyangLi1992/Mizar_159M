import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from metrics.answer_normalization import extract_mcq_option, extract_yes_no, normalize_answer

def get_option(response):
    return extract_mcq_option(response) or "a"

option2number = {
    'a': 1, 
    'b': 2,
    'c': 3,
    'd': 4,
}

def evaluate_entail_metric(preds, answers):
    scores = {
                "ACC":{}, 'Precision': {}, 'Recall': {}, 'F1': {}, 'main':{},
                "ACC_Entailment": {}, "ACC_Neutral": {}, "ACC_Contradiction": {},
              }

    answers = [option2number.get(get_option(x), 1) for x in answers]
    preds = [option2number.get(get_option(x), 1) for x in preds]
    result = pd.DataFrame({"answers": answers, "preds": preds})

    scores["ACC"]["score"] = accuracy_score(answers, preds)
    scores["Precision"]["score"] = precision_score(answers, preds, average="macro")
    scores["Recall"]["score"] = recall_score(answers, preds, average="macro")
    scores["F1"]["score"] = f1_score(answers, preds, average="macro")

    for label, name in [[1,"Entailment"], [2,"Neutral"], [3,"Contradiction"]]:
        df = result[result["answers"] == label]
        acc = accuracy_score(df["answers"], df["preds"])
        scores[f"ACC_{name}"]["score"] = acc
    
    scores["main"]["score"] = scores["ACC"]["score"]
    return scores

def evaluate_metric(preds, answers):
    corr = 0
    scores = {"ACC":{}, 'main':{}}
    # compute metrics
    for i in range(len(preds)):
        answer = answers[i]
        response = preds[i]
        option = get_option(response)
        answer_option = extract_mcq_option(answer)
        if answer_option is not None:
            correct = option == answer_option
        else:
            correct = normalize_answer(response) == normalize_answer(answer)

        if correct:
            corr += 1

    scores["ACC"]['score'] = (corr/len(preds)) * 100
    scores["main"]["score"] = scores["ACC"]['score']
    return scores

def evaluate_metric_binary(preds, answers):
    corr = 0
    scores = {"ACC":{}, 'main':{}}
    # compute metrics
    for i in range(len(preds)):
        answer = answers[i]
        response = preds[i]
        answer_label = extract_yes_no(answer)
        response_label = extract_yes_no(response)
        if answer_label is not None and response_label is not None:
            correct = answer_label == response_label
        else:
            correct = normalize_answer(answer) in normalize_answer(response)

        if correct:
            corr += 1

    scores["ACC"]['score'] = (corr/len(preds)) * 100
    scores["main"]["score"] = scores["ACC"]['score']
    return scores
