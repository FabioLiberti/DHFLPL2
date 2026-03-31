"""Meccanismi di Differential Privacy per Federated Learning.

Implementa (epsilon, delta)-differential privacy come descritto nel paper:
Pr[M(D) in S] <= e^epsilon * Pr[M(D') in S] + delta

Due azioni principali (Sezione 5.2 del paper):
1. Redazione dati privati (email, telefono, indirizzi)
2. Aggiunta di rumore per offuscare valori numerici sensibili

Riferimento: Sezione 4.2, Equazione (4) del paper.
"""

import re
import numpy as np


# --- Redazione dati privati (Azione 1) ---

DEFAULT_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
    "address": r"\d{1,5}\s[\w\s]{2,30}(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|court|ct|via|piazza|viale|corso)",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "codice_fiscale": r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
}


def redact_private_data(text, patterns=None, replacement="[REDACTED]"):
    """Redige dati privati dal testo applicando pattern regex.

    Args:
        text: Stringa di testo da processare.
        patterns: Dict {nome: regex_pattern}. Se None, usa i default.
        replacement: Stringa sostitutiva.

    Returns:
        Tuple (testo_redatto, dict_conteggi) con il testo ripulito
        e il conteggio delle redazioni per tipo.
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    redacted = text
    counts = {}

    for name, pattern in patterns.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        counts[name] = len(matches)
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

    return redacted, counts


def redact_private_fields(record, sensitive_fields):
    """Redige campi sensibili da un record (dict).

    Args:
        record: Dizionario con i dati.
        sensitive_fields: Lista di chiavi da redigere.

    Returns:
        Copia del record con i campi sensibili sostituiti.
    """
    redacted = record.copy()
    for field in sensitive_fields:
        if field in redacted:
            redacted[field] = "[REDACTED]"
    return redacted


# --- Aggiunta rumore (Azione 2) ---

def add_gaussian_noise(data, epsilon, delta, sensitivity=1.0):
    """Aggiunge rumore gaussiano per differential privacy.

    Implementa il meccanismo gaussiano per (epsilon, delta)-DP.
    sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        data: Array numpy con i valori da proteggere.
        epsilon: Budget di privacy (piu' basso = piu' privacy).
        delta: Probabilita' di violazione della privacy.
        sensitivity: Sensibilita' della query (default L2 = 1.0).

    Returns:
        Array numpy con rumore aggiunto.
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=data.shape)
    return data + noise


def add_laplace_noise(data, epsilon, sensitivity=1.0):
    """Aggiunge rumore laplaciano per epsilon-differential privacy.

    Implementa il meccanismo di Laplace per epsilon-DP puro.
    scale = sensitivity / epsilon

    Args:
        data: Array numpy con i valori da proteggere.
        epsilon: Budget di privacy.
        sensitivity: Sensibilita' L1 della query.

    Returns:
        Array numpy con rumore aggiunto.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=data.shape)
    return data + noise


def clip_gradients(gradients, clip_norm):
    """Clipping dei gradienti per limitare la sensibilita'.

    Il gradient clipping e' un prerequisito per la DP nei
    modelli di deep learning: limita il contributo massimo
    di un singolo campione.

    Args:
        gradients: Lista di array numpy (gradienti per layer).
        clip_norm: Norma massima consentita.

    Returns:
        Lista di gradienti clippati.
    """
    total_norm = np.sqrt(
        sum(np.sum(g ** 2) for g in gradients)
    )
    clip_factor = min(1.0, clip_norm / (total_norm + 1e-8))
    return [g * clip_factor for g in gradients]


def apply_dp_to_weights(weights, epsilon, delta, clip_norm=1.0):
    """Applica differential privacy ai pesi del modello.

    Pipeline completa:
    1. Clip dei pesi per limitare la sensibilita'
    2. Aggiunta di rumore gaussiano calibrato

    Args:
        weights: Lista di array numpy (pesi del modello).
        epsilon: Budget di privacy.
        delta: Parametro delta per DP gaussiana.
        clip_norm: Norma massima per il clipping.

    Returns:
        Lista di pesi con DP applicata.
    """
    clipped = clip_gradients(weights, clip_norm)
    noisy_weights = [
        add_gaussian_noise(w, epsilon, delta, sensitivity=clip_norm)
        for w in clipped
    ]
    return noisy_weights
