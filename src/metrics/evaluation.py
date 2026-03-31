"""Metriche di valutazione per il modello federato.

Calcola precision, recall, F1-score e accuracy del modello
aggregato sui dati di test.

Riferimento: Listing 3 del paper.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_model(model, x_test, y_test):
    """Valuta il modello con metriche di classificazione.

    Args:
        model: Modello Keras da valutare.
        x_test: Dati di test.
        y_test: Labels di test.

    Returns:
        Dict con precision, recall, f1, accuracy, loss.

    Riferimento: Listing 3 del paper.
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test.flatten()

    precision = precision_score(
        y_true, y_pred_classes, average="weighted", zero_division=0,
    )
    recall = recall_score(
        y_true, y_pred_classes, average="weighted", zero_division=0,
    )
    f1 = f1_score(
        y_true, y_pred_classes, average="weighted", zero_division=0,
    )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_per_class(model, x_test, y_test, class_names=None):
    """Valutazione dettagliata per classe.

    Args:
        model: Modello Keras da valutare.
        x_test: Dati di test.
        y_test: Labels di test.
        class_names: Lista opzionale di nomi delle classi.

    Returns:
        Dict con metriche per-class e metriche aggregate.
    """
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test.flatten()

    num_classes = y_pred.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    per_class = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        class_pred = (y_pred_classes == i)
        class_true = (y_true == i)
        tp = (class_pred & class_true).sum()
        fp = (class_pred & ~class_true).sum()
        fn = (~class_pred & class_true).sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(mask.sum()),
        }

    return per_class
