# Changelog

Tutte le modifiche rilevanti al progetto DHFLPL2 sono documentate in questo file.

## [0.1.0] - 2026-03-31

### Aggiunto
- Struttura iniziale del repository
- Modulo `src/models/cnn.py`: definizione CNN adattabile ai 5 dataset del paper
- Modulo `src/data/loader.py`: caricamento dataset (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN)
- Modulo `src/data/partitioner.py`: distribuzione non-IID dei dati tra client federati
- Modulo `src/utils/config.py`: configurazione centralizzata
- File di configurazione: requirements.txt, setup.py, .gitignore, LICENSE
- CHANGELOG.md per tracciamento sviluppo
