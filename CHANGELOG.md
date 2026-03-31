# Changelog

Tutte le modifiche rilevanti al progetto DHFLPL2 sono documentate in questo file.

## [0.4.0] - 2026-03-31

### Aggiunto
- Script `experiments/run_experiment.py`: entry point per riprodurre tutti gli esperimenti del paper (5 dataset, 2-50 client, 150 round, supporto DP)
- Configurazioni YAML per ogni dataset: cifar10, cifar100, mnist, fashion_mnist, svhn
- Script `scripts/plot_results.py`: generazione Figure 2 (griglia accuracy/loss), Figure 3 (federato vs centralizzato), tabella riassuntiva

## [0.3.0] - 2026-03-31

### Aggiunto
- Modulo `src/privacy/dp_mechanism.py`: differential privacy con meccanismo gaussiano e laplaciano, redazione dati privati, gradient clipping, pipeline DP sui pesi
- Modulo `src/privacy/threat_model.py`: simulazione 4 vettori di attacco (Gradient Inversion, Model Update Leakage, Membership Inference, Side-Channel Analysis)
- Test unitari: test_privacy.py (redazione, rumore, clipping, analisi rischio)

## [0.2.0] - 2026-03-31

### Aggiunto
- Modulo `src/federation/server.py`: server FedAvg con orchestrazione completa del processo federato
- Modulo `src/federation/client.py`: client FL con training locale (Listing 2 del paper)
- Modulo `src/federation/strategy.py`: aggregazione FedAvg pesata e media semplice (Algorithm 1 del paper)
- Modulo `src/metrics/evaluation.py`: valutazione con precision, recall, F1, accuracy (Listing 3 del paper)
- Test unitari: test_models.py, test_data.py, test_federation.py

## [0.1.0] - 2026-03-31

### Aggiunto
- Struttura iniziale del repository
- Modulo `src/models/cnn.py`: definizione CNN adattabile ai 5 dataset del paper
- Modulo `src/data/loader.py`: caricamento dataset (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN)
- Modulo `src/data/partitioner.py`: distribuzione non-IID dei dati tra client federati
- Modulo `src/utils/config.py`: configurazione centralizzata
- File di configurazione: requirements.txt, setup.py, .gitignore, LICENSE
- CHANGELOG.md per tracciamento sviluppo
