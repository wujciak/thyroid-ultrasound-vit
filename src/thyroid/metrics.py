"""Image- and patient-level metrics for a binary malignancy score."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity; chosen on validation, reused on test."""
    fpr, tpr, thr = roc_curve(y, score)
    return float(thr[np.argmax(tpr - fpr)])


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    tp = np.sum(pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    sensitivity = tp / max(np.sum(y == 1), 1)
    specificity = tn / max(np.sum(y == 0), 1)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "accuracy": float(np.mean(pred == y)),
        "majority_accuracy": float(max(np.mean(y), 1 - np.mean(y))),
    }


def patient_scores(patient_ids: list[str], y: np.ndarray, score: np.ndarray):
    """Mean image score per patient: the unit a clinician actually decides on."""
    ids = np.array(patient_ids)
    unique = sorted(set(patient_ids))
    p_y = np.array([y[ids == p][0] for p in unique])
    p_score = np.array([score[ids == p].mean() for p in unique])
    return unique, p_y, p_score


def thresholds(patient_ids: list[str], y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    _, p_y, p_score = patient_scores(patient_ids, y, score)
    return {"image": youden_threshold(y, score), "patient": youden_threshold(p_y, p_score)}


def evaluate(patient_ids: list[str], y: np.ndarray, score: np.ndarray, thr: dict[str, float]) -> dict:
    _, p_y, p_score = patient_scores(patient_ids, y, score)
    return {
        "image": binary_metrics(y, score, thr["image"]),
        "patient": binary_metrics(p_y, p_score, thr["patient"]),
        "thresholds": thr,
    }
