# Changelog

Tutte le modifiche rilevanti al progetto DHFLPL2 sono documentate in questo file.

## [1.0.2] - 2026-03-31

### Aggiunto
- `environment.yml`: Conda environment con tutte le dipendenze (Python 3.11, TensorFlow, Flower, pydp)
- README aggiornato con istruzioni di installazione Conda

## [1.0.1] - 2026-03-31

### Corretto
- Rinominata `evaluate_model()` -> `evaluate_with_metrics()` per allineamento esatto con Listing 3 del paper
- Aggiunta curva "Similar Approach" [29] (Shamsian et al.) in Figure 3, coerente con il paper
- Aggiunta modalita' `--mode flower` al runner per deployment distribuito via Flower

### Aggiunto
- Config YAML con Differential Privacy abilitata per tutti i 5 dataset (`*_dp.yml`)

## [1.0.0] - 2026-03-31

### Aggiunto
- README completo: overview, quick start, architettura, struttura progetto, istruzioni Docker e k3s
- Test di integrazione end-to-end (`tests/test_integration.py`): pipeline FL completa, FL con DP, metriche, redazione, analisi minacce, coerenza configurazione col paper

### Modificato
- README riscritto con documentazione tecnica completa, comandi di utilizzo e deployment

## [0.5.0] - 2026-03-31

### Aggiunto
- `deploy/docker/Dockerfile.server`: container per Flower SuperLink (aggregatore)
- `deploy/docker/Dockerfile.client`: container per Flower SuperNode (worker)
- `deploy/docker/docker-compose.yml`: composizione locale per test con 2 client
- `deploy/k8s/namespace.yml`: namespace Kubernetes per il sistema FL
- `deploy/k8s/server-deployment.yml`: deployment + service per il server FL
- `deploy/k8s/client-deployment.yml`: deployment scalabile per i client FL (2-50 repliche)
- `deploy/k8s/autoscaler.yml`: custom autoscaler basato su precision (CronJob + RBAC)
- `src/federation/server_app.py`: entry point Flower server per container
- `src/federation/client_app.py`: entry point Flower client per container con supporto DP
- `scripts/setup_cluster.sh`: setup k3s controller/worker + Rancher
- `scripts/deploy.sh`: build immagini + deploy/scale su k3s

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
