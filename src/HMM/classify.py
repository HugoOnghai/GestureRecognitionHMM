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

    best_model_name = None
    best_prob = -np.inf

    for model in models:
        loglikelihood, termination_prob = model.score(seq)
        if termination_prob > best_prob:
            best_prob = termination_prob
            best_model_name = model.label
    
    return best_model_name, best_prob
