"""Demo: Data Redaction (Privacy dei dati personali).

Dimostra la redazione dei dati privati come descritto nel paper:
1. Redazione di email, telefoni, indirizzi da testo
2. Redazione di campi sensibili da record strutturati
3. Aggiunta di rumore a valori numerici sensibili

Riferimento: Sezione 5.2 del paper - "redacted the private data
of the users, such as photos tagged as personal and data such as
email addresses, phone numbers, home addresses".

Uso:
    python -m demos.demo_data_redaction
"""

import os

import numpy as np

from src.privacy.dp_mechanism import (
    redact_private_data,
    redact_private_fields,
    add_gaussian_noise,
    add_laplace_noise,
)


def demo_text_redaction():
    """Demo di redazione dati da testo libero."""
    print("=" * 60)
    print("1. REDAZIONE DATI DA TESTO")
    print("=" * 60)

    samples = [
        "Il paziente Mario Rossi (mario.rossi@email.com) ha chiamato "
        "il numero +39 333-456-7890 per prenotare una visita. "
        "Risiede in Via Roma 42 street.",

        "Fattura per utente con codice fiscale RSSMRA80A01H501Z, "
        "carta di credito 4111-2222-3333-4444, "
        "contatto: info@hospital.it, tel. 06 1234567.",

        "Record medico: paziente John Doe, email john.doe@clinic.com, "
        "telefono +1 555-987-6543, indirizzo 123 Main street.",
    ]

    for i, text in enumerate(samples):
        print(f"\n--- Esempio {i+1} ---")
        print(f"ORIGINALE:\n  {text}")
        redacted, counts = redact_private_data(text)
        print(f"\nREDATTO:\n  {redacted}")
        print(f"\nConteggio redazioni: {counts}")


def demo_field_redaction():
    """Demo di redazione campi sensibili da record strutturati."""
    print(f"\n{'='*60}")
    print("2. REDAZIONE CAMPI DA RECORD")
    print("=" * 60)

    records = [
        {
            "patient_id": "P001",
            "name": "Mario Rossi",
            "email": "mario.rossi@hospital.it",
            "phone": "+39 333 1234567",
            "diagnosis": "Ipertensione",
            "age": 45,
        },
        {
            "patient_id": "P002",
            "name": "Laura Bianchi",
            "email": "l.bianchi@clinic.com",
            "phone": "+39 340 9876543",
            "address": "Via Garibaldi 15, Roma",
            "diagnosis": "Diabete tipo 2",
            "age": 62,
        },
    ]

    sensitive_fields = ["name", "email", "phone", "address"]

    for i, record in enumerate(records):
        print(f"\n--- Record {i+1} ---")
        print(f"ORIGINALE:  {record}")
        redacted = redact_private_fields(record, sensitive_fields)
        print(f"REDATTO:    {redacted}")


def demo_numerical_noise():
    """Demo di aggiunta rumore a valori numerici sensibili."""
    print(f"\n{'='*60}")
    print("3. AGGIUNTA RUMORE A VALORI NUMERICI")
    print("=" * 60)

    print("\nScenario: protezione di valori medici sensibili")

    original_values = np.array([120.0, 80.0, 98.6, 72.0, 5.8])
    labels = ["Pressione Sist.", "Pressione Diast.", "Temperatura",
              "Battito Cardiaco", "Glicemia (mmol/L)"]

    print(f"\n{'Metrica':<20} {'Originale':>10} {'Gauss(eps=1)':>14} "
          f"{'Gauss(eps=5)':>14} {'Laplace(eps=1)':>16}")
    print("-" * 76)

    noisy_gauss_1 = add_gaussian_noise(
        original_values, epsilon=1.0, delta=1e-5, sensitivity=10.0,
    )
    noisy_gauss_5 = add_gaussian_noise(
        original_values, epsilon=5.0, delta=1e-5, sensitivity=10.0,
    )
    noisy_laplace = add_laplace_noise(
        original_values, epsilon=1.0, sensitivity=10.0,
    )

    for i, label in enumerate(labels):
        print(f"{label:<20} {original_values[i]:>10.1f} "
              f"{noisy_gauss_1[i]:>14.1f} "
              f"{noisy_gauss_5[i]:>14.1f} "
              f"{noisy_laplace[i]:>16.1f}")

    print(f"\n{'Errore medio':<20} {'':>10} "
          f"{np.mean(np.abs(noisy_gauss_1 - original_values)):>14.2f} "
          f"{np.mean(np.abs(noisy_gauss_5 - original_values)):>14.2f} "
          f"{np.mean(np.abs(noisy_laplace - original_values)):>16.2f}")

    print("\nNota: epsilon piu' alto = meno rumore = meno privacy")
    print("      epsilon piu' basso = piu' rumore = piu' privacy")


def main():
    print(f"{'='*60}")
    print(f"DEMO: Data Redaction e Noise Augmentation")
    print(f"Riferimento: Sezione 5.2 del paper")
    print(f"{'='*60}")

    demo_text_redaction()
    demo_field_redaction()
    demo_numerical_noise()

    print(f"\n{'='*60}")
    print("DEMO COMPLETATA")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
