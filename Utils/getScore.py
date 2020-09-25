import json 
import os 
from preprocessData import CoQAExample
import pickle
import string 
import re
from collections import Counter

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0

    if len(answers) == 0:
        return 1. if len(pred) == 0 else 0.
    
    g_tokens = _normalize_answer(pred).split()
    ans_tokens = [_normalize_answer(answer).split() for answer in answers]
    scores = [_score(g_tokens, a) for a in ans_tokens]
    if len(ans_tokens) == 1:
        score = scores[0]
    else:
        score = 0
        for i in range(len(ans_tokens)):
            scores_one_out = scores[:i] + scores[(i + 1):]
            score += max(scores_one_out)
        score /= len(ans_tokens)
    return score

def score(pred, truth):
    assert len(pred) == len(truth)
    no_ans_total = no_total = yes_total = normal_total = total = 0
    no_ans_f1 = no_f1 = yes_f1 = normal_f1 = f1 = 0
    all_f1s = []
    for p, t in zip(pred, truth):
        total += 1
        this_f1 = _f1_score(p, t)
        f1 += this_f1
        all_f1s.append(this_f1)
        if t[0].lower() == 'no':
            no_total += 1
            no_f1 += this_f1
        elif t[0].lower() == 'yes':
            yes_total += 1
            yes_f1 += this_f1
        elif t[0].lower() == 'unknown':
            no_ans_total += 1
            no_ans_f1 += this_f1
        else:
            normal_total += 1
            normal_f1 += this_f1

    f1 = 100. * f1 / total
    if no_total == 0:
        no_f1 = 0.
    else:
        no_f1 = 100. * no_f1 / no_total
    if yes_total == 0:
        yes_f1 = 0
    else:
        yes_f1 = 100. * yes_f1 / yes_total
    if no_ans_total == 0:
        no_ans_f1 = 0.
    else:
        no_ans_f1 = 100. * no_ans_f1 / no_ans_total
    normal_f1 = 100. * normal_f1 / normal_total
    result = {
        'total': total,
        'f1': f1,
        'no_total': no_total,
        'no_f1': no_f1,
        'yes_total': yes_total,
        'yes_f1': yes_f1,
        'no_ans_total': no_ans_total,
        'no_ans_f1': no_ans_f1,
        'normal_total': normal_total,
        'normal_f1': normal_f1,
    }
    return result, all_f1s

with open('preprocess/gold_answers.pkl', 'rb') as f:
    gold_answers = pickle.load(f)

files = os.listdir('predictions/')

for file in files:
    if(file[0] == '.'):
        continue
    with open('predictions/'+file, 'r') as f:
        preds = json.load(f)
    predictions = []
    for _, pred in preds.items():
        if(pred == "unkownn"):
            pred = "unknown"
        predictions.append(pred)
    result, _ = score(predictions, gold_answers)
    print(file, result['f1'])
    
    