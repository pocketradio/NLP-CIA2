"""
metrics.py
Evaluation metrics for sentence ordering: pairwise accuracy,
sequence accuracy, Kendall Tau, and helper utilities.
"""

import numpy as np
from scipy.stats import kendalltau as scipy_kendalltau


def pairwise_accuracy(pred_pairs, true_pairs):
    """
    pred_pairs: list of (i, j, label) where label=1 means i before j
    true_pairs: same format
    Returns fraction of pairs where prediction matches ground truth.
    """
    if not true_pairs:
        return 0.0
    correct = sum(
        p == t
        for (_, _, p), (_, _, t) in zip(pred_pairs, true_pairs)
    )
    return correct / len(true_pairs)


def sequence_accuracy(pred_orders, true_orders):
    """
    pred_orders: list of lists (predicted sentence indices)
    true_orders: list of lists (true sentence indices)
    Returns fraction of documents where full predicted order matches true order exactly.
    """
    if not true_orders:
        return 0.0
    correct = sum(
        list(p) == list(t)
        for p, t in zip(pred_orders, true_orders)
    )
    return correct / len(true_orders)


def kendall_tau(pred_order, true_order):
    """Kendall Tau correlation between predicted and true orderings."""
    if len(pred_order) <= 1:
        return 1.0
    tau, _ = scipy_kendalltau(pred_order, true_order)
    if np.isnan(tau):
        return 0.0
    return float(tau)


def mean_kendall_tau(pred_orders, true_orders):
    """Mean Kendall Tau across all documents."""
    taus = [
        kendall_tau(p, t)
        for p, t in zip(pred_orders, true_orders)
        if len(p) > 1
    ]
    return float(np.mean(taus)) if taus else 0.0


def build_pairwise_labels(n_sentences):
    """
    Return list of (i, j, label) for all ordered pairs i != j.
    label=1 if i < j (i comes before j in canonical order), else 0.
    """
    pairs = []
    for i in range(n_sentences):
        for j in range(n_sentences):
            if i != j:
                pairs.append((i, j, 1 if i < j else 0))
    return pairs


def tournament_to_order(score_matrix):
    """
    Convert pairwise score matrix to a total ordering.
    score_matrix[i][j] = probability/score that sentence i comes before j.
    Returns: list of sentence indices sorted by descending row-sum score.
    """
    n = score_matrix.shape[0]
    row_scores = score_matrix.sum(axis=1)
    return list(np.argsort(-row_scores))
