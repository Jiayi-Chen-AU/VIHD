import torch
import numpy as np
import pdb

# for sentence-level semantic entropy
from SeEntLib.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from SeEntLib.uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from SeEntLib.uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id_pt
from SeEntLib.uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao_pt


def get_sentence_semantic_entropy_w_semantic_ids(log_liks, semantic_ids):
    # Length normalization of generation probabilities.
    log_liks_agg = [log_lik.mean() for log_lik in log_liks]

    # Compute semantic entropy.
    log_likelihood_per_semantic_id = logsumexp_by_id_pt(semantic_ids, torch.stack(log_liks_agg, dim=0), agg='sum_normalized')
    semantic_entropy = predictive_entropy_rao_pt(torch.stack(log_likelihood_per_semantic_id, dim=0))
    # record
    prob_per_semantic_id = torch.exp(torch.stack(log_likelihood_per_semantic_id, dim=0))
    return semantic_entropy, prob_per_semantic_id

def get_sentence_semantic_distribution(scores, semantic_ids):
    # Length normalization of generation probabilities.
    score_agg = [score.mean() for score in scores]
    score_agg = torch.stack(score_agg, dim=0)

    semantic_ids = torch.tensor(semantic_ids, device=score_agg.device)
    unique_ids = torch.unique(semantic_ids)
    score_per_semantic_id = []
    
    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        mask = (semantic_ids == uid)
        # Gather log likelihoods at these indices.
        id_score = score_agg[mask]
        score_per_semantic_id.append(id_score.sum())

    score_per_semantic_id = torch.stack(score_per_semantic_id, dim=0)
    return score_per_semantic_id


def get_sentence_semantic_entropy(question, responses, log_liks, entailment_model):
    # log_liks: Token log likelihoods
    responses = [f'{question} {r}' for r in responses]
    semantic_ids = get_semantic_ids(responses, model=entailment_model, strict_entailment=True)

    # Length normalization of generation probabilities.
    log_liks_agg = [log_lik.mean() for log_lik in log_liks]

    # Compute semantic entropy.
    log_likelihood_per_semantic_id = logsumexp_by_id_pt(semantic_ids, torch.stack(log_liks_agg, dim=0), agg='sum_normalized')
    semantic_entropy = predictive_entropy_rao_pt(torch.stack(log_likelihood_per_semantic_id, dim=0))

    # record
    prob_per_semantic_id = torch.exp(torch.stack(log_likelihood_per_semantic_id, dim=0))
    return semantic_entropy, semantic_ids, prob_per_semantic_id

