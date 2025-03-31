import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def load_binary_gold_label(gold_label_transform, avg_formal, votes_formal):
    #We transform the labels available to us to a binary label (formal or informal)
    if gold_label_transform == "avg_sign":
        #Check whether the average of all votes is positive (formal) or negative (informal)
        if avg_formal >= 0: 
            return 1
        else:
            return 0
    elif gold_label_transform == "maj_sign":
        #Check all votes; if most of them are positive (formal), then we predicte formal, and vice versa
        votes_pos = [1 if item>0 else 0 for item in votes_formal]
        num_votes_pos = sum(votes_pos)
        votes_neg = [1 if item<0 else 0 for item in votes_formal]
        num_votes_neg = sum(votes_neg)

        if num_votes_pos >= num_votes_neg:
            return 1
        else:
            return 0
    elif gold_label_transform == "sample_sign":
        # Randomly choose one of the votes and check its sign
        vote = random.sample(votes_formal, 1)[0]
        if vote >=0:
            return 1
        else:
            return 0
    else:
        raise ValueError(f"Unknown technique for transforming data to binary result: {gold_label_transform}")

def compute_metrics(preds, gold_labels):
    f1 = f1_score(gold_labels, preds)
    precision = precision_score(gold_labels, preds)
    recall = recall_score(gold_labels, preds)
    accuracy = accuracy_score(gold_labels, preds)

    perforamance_metrics = {"f1":f1, "precision":precision, "recall":recall, "accuracy":accuracy}
    return perforamance_metrics
