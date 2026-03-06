from src.HMM.HMM import HMM
import numpy as np

def classify(seq, models):
    """
    seq is a sequence of observations,
    models is a list of HMM objects, one for each possible gesture type
    or models is a dict of HMM objects, one for each possible gesture type, which we convert to a list
    """

    if isinstance(models, dict):
        models = list(models.values())

    scores = []

    for model in models:
        ll = model.score(seq)
        scores.append((model.label, ll))
        
    # sort by log-likelihood descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[0][0], scores[0][1], scores
