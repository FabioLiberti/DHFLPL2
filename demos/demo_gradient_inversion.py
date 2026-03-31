"""Demo: Gradient Inversion Attack.

Simula un attaccante che tenta di ricostruire i dati originali
dai gradienti condivisi durante il training federato.

Mostra:
1. Immagine originale del client
2. Immagine ricostruita dall'attaccante (senza DP)
3. Immagine ricostruita dall'attaccante (con DP)
4. Confronto visivo: la DP rende la ricostruzione inefficace

Riferimento: Sezione 4.2 del paper - Gradient Inversion Attacks.

Uso:
    python -m demos.demo_gradient_inversion
    python -m demos.demo_gradient_inversion --epsilon 0.5
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.cnn import create_model
from src.data.loader import load_dataset
from src.privacy.dp_mechanism import add_gaussian_noise


def compute_gradients(model, x_sample, y_sample):
    """Calcola i gradienti del modello su un singolo campione."""
    x = tf.constant(x_sample[np.newaxis], dtype=tf.float32)
    y = tf.constant([y_sample], dtype=tf.int64)

    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    return [g.numpy() for g in gradients]


def gradient_inversion_attack(model, target_gradients, input_shape,
                              num_iterations=500, lr=0.05):
    """Tenta di ricostruire l'input dai gradienti osservati.

    L'attaccante parte da un input casuale e lo ottimizza
    per minimizzare la distanza tra i gradienti calcolati
    sull'input fittizio e quelli osservati.
    """
    dummy = tf.Variable(
        tf.random.normal([1, *input_shape], stddev=0.1),
        dtype=tf.float32,
    )
    dummy_label = tf.Variable(tf.zeros([1, 10]), dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_history = []

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(dummy)
            predictions = model(dummy, training=True)
            dummy_loss = tf.keras.losses.sparse_categorical_crossentropy(
                tf.argmax(predictions, axis=1), predictions,
            )
            dummy_grads = tape.gradient(dummy_loss, model.trainable_variables)

            grad_diff = sum(
                tf.reduce_sum((dg - tf.constant(tg)) ** 2)
                for dg, tg in zip(dummy_grads, target_gradients)
                if dg is not None
            )

        grad_of_dummy = tape.gradient(grad_diff, dummy)
        if grad_of_dummy is not None:
            optimizer.apply_gradients([(grad_of_dummy, dummy)])

        loss_history.append(float(grad_diff))

    reconstructed = tf.clip_by_value(dummy, 0, 1).numpy()[0]
    return reconstructed, loss_history


def plot_results(original, recon_no_dp, recon_with_dp,
                 loss_no_dp, loss_with_dp, epsilon, output_path):
    """Genera il pannello di confronto."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Riga 1: immagini
    cmap = "gray" if original.shape[-1] == 1 else None
    img_orig = original.squeeze() if original.shape[-1] == 1 else original
    img_no_dp = recon_no_dp.squeeze() if recon_no_dp.shape[-1] == 1 else recon_no_dp
    img_dp = recon_with_dp.squeeze() if recon_with_dp.shape[-1] == 1 else recon_with_dp

    axes[0, 0].imshow(img_orig, cmap=cmap)
    axes[0, 0].set_title("Immagine Originale\n(dato del client)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.clip(img_no_dp, 0, 1), cmap=cmap)
    axes[0, 1].set_title("Ricostruzione SENZA DP\n(attacco efficace)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.clip(img_dp, 0, 1), cmap=cmap)
    axes[0, 2].set_title(f"Ricostruzione CON DP (eps={epsilon})\n(attacco mitigato)",
                         fontsize=11)
    axes[0, 2].axis("off")

    # Riga 2: loss di ricostruzione + analisi
    axes[1, 0].plot(loss_no_dp, linewidth=0.8)
    axes[1, 0].set_xlabel("Iterazione")
    axes[1, 0].set_ylabel("Gradient Distance")
    axes[1, 0].set_title("Convergenza attacco\n(senza DP)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(loss_with_dp, linewidth=0.8, color="orange")
    axes[1, 1].set_xlabel("Iterazione")
    axes[1, 1].set_ylabel("Gradient Distance")
    axes[1, 1].set_title("Convergenza attacco\n(con DP)")
    axes[1, 1].grid(True, alpha=0.3)

    # MSE confronto
    mse_no_dp = np.mean((original - recon_no_dp) ** 2)
    mse_dp = np.mean((original - recon_with_dp) ** 2)
    labels = ["Senza DP", f"Con DP (eps={epsilon})"]
    values = [mse_no_dp, mse_dp]
    colors = ["#e74c3c", "#2ecc71"]
    axes[1, 2].bar(labels, values, color=colors)
    axes[1, 2].set_ylabel("MSE (errore ricostruzione)")
    axes[1, 2].set_title("Errore di ricostruzione\n(piu' alto = piu' sicuro)")
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Gradient Inversion Attack: effetto della Differential Privacy",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGrafico salvato in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Gradient Inversion Attack con/senza DP",
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"DEMO: Gradient Inversion Attack")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Iterazioni attacco: {args.iterations}")
    print(f"{'='*60}")

    print("\nCaricamento MNIST...")
    (x_train, y_train), _ = load_dataset("mnist")

    input_shape = (28, 28, 1)
    model = create_model(input_shape, 10)

    # Seleziona un campione
    idx = 42
    x_sample = x_train[idx]
    y_sample = y_train[idx]
    print(f"Campione selezionato: indice {idx}, label {y_sample}")

    # Gradienti reali (senza DP)
    print("\nCalcolo gradienti senza DP...")
    grads_clean = compute_gradients(model, x_sample, y_sample)

    # Gradienti con DP
    print("Calcolo gradienti con DP...")
    grads_dp = [
        add_gaussian_noise(g, epsilon=args.epsilon, delta=1e-5)
        for g in grads_clean
    ]

    # Attacco senza DP
    print(f"\nAttacco su gradienti SENZA DP ({args.iterations} iterazioni)...")
    recon_no_dp, loss_no_dp = gradient_inversion_attack(
        model, grads_clean, input_shape, num_iterations=args.iterations,
    )

    # Attacco con DP
    print(f"Attacco su gradienti CON DP ({args.iterations} iterazioni)...")
    recon_dp, loss_dp = gradient_inversion_attack(
        model, grads_dp, input_shape, num_iterations=args.iterations,
    )

    mse_no_dp = np.mean((x_sample - recon_no_dp) ** 2)
    mse_dp = np.mean((x_sample - recon_dp) ** 2)

    print(f"\n{'='*60}")
    print(f"RISULTATI")
    print(f"  MSE senza DP: {mse_no_dp:.6f} (ricostruzione migliore = meno sicuro)")
    print(f"  MSE con DP:   {mse_dp:.6f} (ricostruzione peggiore = piu' sicuro)")
    print(f"  Rapporto:     {mse_dp/mse_no_dp:.1f}x piu' difficile con DP")
    print(f"{'='*60}")

    plot_results(
        x_sample, recon_no_dp, recon_dp,
        loss_no_dp, loss_dp, args.epsilon,
        os.path.join(output_dir, "gradient_inversion_attack.png"),
    )


if __name__ == "__main__":
    main()
