# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FLAPS** (Federated Learning Acoustic Perception and Separation) — a music source separation system deployed via federated learning across edge devices (Jetson/RPi nodes on an 802.11s mesh network).

The codebase lives under `MobFedLS/`, which integrates a custom ML application with the **MobFedLearnSys (MFLS)** federated learning framework. Several components are git submodules pointing to the `ai4sme/MobFedLearnSys` upstream.

## Commands

### Docker (primary workflow)

```bash
# Build the FLAPS ML app container
docker build -t ml-app-flaps -f MobFedLS/tools/ml-apps/flaps/deployment/Dockerfile MobFedLS/tools/ml-apps/flaps/

# Run single ML app (standalone mode)
cd MobFedLS/tools/ml-apps/flaps && docker compose up --build

# Run full federated setup (2 clients + flower server + ghost clients)
cd MobFedLS/tools/ml-apps/flaps && docker compose -f docker-compose.federated.yaml up
# Then start training:
docker compose -f docker-compose.federated.yaml --profile training up

# Convenience wrapper with configurable FL params
./MobFedLS/tools/ml-apps/flaps/start-training.sh
```

### Local Python

```bash
# Install dependencies
pip install -r MobFedLS/tools/ml-apps/flaps/requirements-mlapp.txt

# Standalone training
python MobFedLS/tools/ml-apps/flaps/scripts/train_musdb.py --config scripts/train_musdb.example.json

# Inference from audio file
python MobFedLS/tools/ml-apps/flaps/scripts/predict_from_file.py

# Real-time mic inference
python MobFedLS/tools/ml-apps/flaps/scripts/predict_from_mic.py
```

### Tests

```bash
# Run all tests
pytest MobFedLS/tools/ml-apps/flaps/tests/

# Run a single test file
pytest MobFedLS/tools/ml-apps/flaps/tests/test_model_shapes.py
```

## Architecture

### Component Map

```
flower-server  ←gRPC→  flower-ghostclient-{1,2}  ←HTTP→  ml-app-{1,2}
                                                               ↑
                                                        mfls-interface
                                                        (hook contract)
```

- **`MobFedLS/tools/ml-apps/flaps/`** — the FLAPS application (Python/PyTorch). The code split:
  - `cmd/clientML.py` — MobFedLS hook entry points (`get_parameters`, `set_parameters`, `fit`, `evaluate`, `predict`). This is where the framework calls into the ML app.
  - `cmd/ml_app/mfls_app.py` — `MLApp` class owning the training loop, dataset, model, and optimizer state.
  - `cmd/ml_app/models/` — `UNetSmall` and `UNetLarge` spectrogram-domain U-Nets.
  - `cmd/ml_app/utils/audio.py` — STFT/iSTFT helpers and SI-SDR metric.
  - `cmd/ml_app/utils/losses.py` — Composite loss: mask + magnitude + consistency + KL + SI-SDR.
  - `cmd/ml_app/dataset/musdb_loader.py` — `MusdbSeparationDataset` with manifest building and hash-based client partitioning.

- **`MobFedLS/tools/flower-ghostclient/`** (submodule) — gRPC↔HTTP bridge; translates Flower protocol calls into HTTP requests to the ML app's Flask endpoint.

- **`MobFedLS/tools/flower-server/`** (submodule) — Flower FedAvg aggregation server.

- **`MobFedLS/internal/mfls-interface/`** (submodule) — Defines the hook interface contract that `clientML.py` must implement.

- **`MobFedLS/cmd/`** (submodules) — `find-neighbours` (mesh peer discovery) and `mfls-manager` (lifecycle orchestration). Not yet wired to FLAPS.

### Federated Data Partitioning

Clients receive non-overlapping MUSDB18HQ track subsets via deterministic hash partitioning: `hash(track_name) % NUM_CLIENTS == CLIENT_ID`. Both `CLIENT_ID` and `NUM_CLIENTS` are injected as environment variables at container startup.

### Key Configuration

All tuneable parameters are environment variables. Important ones:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `unet_small` | `unet_small` or `unet_large` |
| `N_ROUNDS` | 3 | Federated rounds |
| `N_EPOCHS` | 1 | Local epochs per round |
| `CLIENT_ID` | — | Zero-based client index |
| `DATA_ROOT` | `/app/dataset` | MUSDB18HQ mount path |
| `SAMPLE_RATE` | 8000 | Target sample rate |
| `USE_AMP` | false | Mixed-precision training |
| `AUGMENT_TRAIN` | false | Enable data augmentation |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `cuda:0` |

Full parameter reference: `MobFedLS/docs/flaps/training-params.md`

### Submodule Workflow

Most framework components (`flower-server`, `flower-ghostclient`, `mfls-interface`, `mfls-common`, `find-neighbours`, `mfls-manager`) are git submodules from the `ai4sme/MobFedLearnSys` upstream. Edit only the FLAPS app (`MobFedLS/tools/ml-apps/flaps/`) directly; changes to submodules must go through their own repos.

### Dataset

MUSDB18HQ is expected at `MobFedLS/assets/datasets/musdb18hq/` for local runs and mounted at `/app/dataset` inside containers. The dataset has 4 stems per track: `drums`, `bass`, `vocals`, `other`. The separation target during training is configurable but defaults to all stems.