# FLAPS — Federated Learning Acoustic Perception and Separation

A distributed music source separation system deployed via Federated Learning across edge devices (Jetson nodes on an 802.11s mesh network). Each node specialises on one instrument stem; FedAvg aggregation at the end of each round combines the specialisations into a single global model without centralising raw audio data.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Machine Learning Core](#2-machine-learning-core)
3. [Federated Learning Integration](#3-federated-learning-integration)
4. [HTTP API Reference](#4-http-api-reference)
5. [Dashboard](#5-dashboard)
6. [Configuration Reference](#6-configuration-reference)
7. [Deployment](#7-deployment)
8. [Local Development](#8-local-development)
9. [Test Suite](#9-test-suite)
10. [Project Structure](#10-project-structure)
11. [Known Limitations / To Do](#11-known-limitations--to-do)

---

## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Jetson 638  (aggregator — 192.168.100.3)                                │
│                                                                          │
│   flower-server ◄──gRPC──►  flower-ghostclient-0 ──HTTP──► ml-app Jetson │
│                             flower-ghostclient-1 ──HTTP──► ml-app Jetson │
│   dashboard (port 8000)                                                  │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐  ┌─────────────────────────────────┐
│  Jetson 681  (client 0)             │  │  Jetson 688  (client 1)         │
│  192.168.100.2                      │  │  192.168.100.1                  │
│  ml-app-flaps  port 5000            │  │  ml-app-flaps  port 5000        │
│  TARGET_STEM=vocals                 │  │  TARGET_STEM=drums              │
└─────────────────────────────────────┘  └─────────────────────────────────┘
```

### Component roles

| Component            | Location           | Purpose                                                                                                 |
|----------------------|--------------------|---------------------------------------------------------------------------------------------------------|
| `ml-app`             | each client Jetson | PyTorch training, inference, playback — exposes a FastAPI server on port 5000                           |
| `flower-ghostclient` | each client Jetson | gRPC↔HTTP bridge; translates Flower protocol calls into HTTP requests to the ml-app                     |
| `flower-server`      | aggregator Jetson  | Flower FedAvg aggregation server; pulls initial parameters from one ml-app, then drives rounds          |
| `dashboard`          | aggregator Jetson  | FastAPI + vanilla-JS web UI on port 8000; manages nodes, audio library, model library, training monitor |

---

## 2. Machine Learning Core

### Models

Two spectrogram-domain U-Net variants, both with a **dual output head**:

| Class       | File                              | Default `base_filters` | Parameters |
|-------------|-----------------------------------|------------------------|------------|
| `UNetSmall` | `cmd/ml_app/models/unet_small.py` | 8                      | ~118 k     |
| `UNetLarge` | `cmd/ml_app/models/unet_large.py` | 24                     | ~1 M       |

**Forward pass** returns `(masks, cls_logits)`:

- `masks` — shape `(B, 4, F, T)` — soft separation masks for each stem (vocals, drums, bass, other)
- `cls_logits` — shape `(B, 4)` — stem presence logits from the classification head (bottleneck → GAP → Linear)

The classification head shares the encoder/bottleneck with the separation head, so it adds **no extra parameters** to the FedAvg-aggregated weight vector.

Select the model via `MODEL_NAME=unet_small` or `MODEL_NAME=unet_large`. Fine-tune capacity with `BASE_FILTERS`.

### Dataset

`MusdbSeparationDataset` (`cmd/ml_app/dataset/musdb_loader.py`) expects MUSDB18HQ at `DATA_ROOT` (container: `/app/dataset`).

The dataset used in this project can be found [here](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)

**Non-IID partitioning** — clients receive non-overlapping track subsets:

```py
partition = hash(track_name) % NUM_CLIENTS == CLIENT_ID
```

Both `CLIENT_ID` and `NUM_CLIENTS` are injected as environment variables at container startup. Setting `FULL_TRACKS=true` bypasses partitioning so every client trains on the full dataset.
Using this parameter introduces a lot of unpredicatability when it comes to VRAM usage. Use with caution.

Per-track segments are extracted with a sliding window of length `SEGMENT_SECONDS` and overlap `TRAIN_CHUNK_OVERLAP`.
Each track has four stems (already in separate files): `drums`, `bass`, `vocals`, `other`.

**Manifest format** — segment offsets (`start_sample`, `num_samples`) are stored in the **source file's native sample rate** (e.g. 44100 Hz), not the model's target rate, so that `soundfile` seeks and reads the correct byte range before resampling. Manifests are cached under `MANIFEST_CACHE_DIR` with a `_v2` filename suffix; old `_v1` manifests (stored at target-rate offsets) are automatically ignored and rebuilt on first run.

**Augmentation** (enabled via `AUGMENT_TRAIN=true`):

| Augmentation   | Env var                     | Default |
|----------------|-----------------------------|---------|
| Random gain    | `AUGMENT_GAIN_DB`           | ±6 dB   |
| Gaussian noise | `AUGMENT_NOISE_STD`         | 0 (off) |
| Phase flip     | `AUGMENT_PHASE_FLIP`        | false   |
| Stem dropout   | `AUGMENT_STEM_DROPOUT_PROB` | 0.3     |
| Stem remix     | `AUGMENT_REMIX_PROB`        | 0 (off) |

Stems zeroed by augmentation are automatically labelled as absent for the classification task.

### Training loop

`MLApp` (`cmd/ml_app/mfls_app.py`) owns the full training lifecycle:

- **Optimiser** — Adam (lr `LEARNING_RATE`, weight decay `WEIGHT_DECAY`)
- **LR scheduler** — `ReduceLROnPlateau` (factor 0.5, patience 3, min_lr 1e-6)
- **Mixed precision** — via `torch.cuda.amp.GradScaler` when `USE_AMP=true`
- **Checkpoint saving** — best checkpoint saved per round to `/app/logs`

### Loss function

`CompositeSourceSeparationLoss` (`cmd/ml_app/utils/losses.py`) — weighted sum of:

| Component | Description |
|---|---|
| Mask BCE | binary cross-entropy on the soft mask vs a hard ideal binary mask |
| Magnitude L1 | L1 on separated magnitude vs reference magnitude |
| Mixture consistency | L1 between sum of separated magnitudes and original mixture magnitude |
| KL regularisation | KL divergence to encourage mask sparsity |
| SI-SDR | scale-invariant signal-to-distortion ratio (direct waveform metric) |
| Classification BCE | per-stem binary cross-entropy on presence logits, weighted by `LOSS_CLS_WEIGHT` |

Only the loss components for the client's own `TARGET_STEM` channel are active during training, so each client specialises while still outputting all four stems.

### Evaluation metrics

Reported per FL round via `fit` and `evaluate` hooks:

- Per-stem SI-SDR and SI-SDR improvement (SI-SDRi) averaged over eval segments
- Mean SI-SDR and mean SI-SDRi across all stems
- Spectral losses (mask BCE, magnitude L1, consistency, KL)
- Classification accuracy (`cls_accuracy`) per stem

---

## 3. Federated Learning Integration

### Hook contract

`cmd/clientML.py` implements the five MobFedLS hooks that `flower-ghostclient` calls over HTTP:

| Hook | Description |
|---|---|
| `get_parameters` | Serialise current model weights as a flat list of NumPy arrays |
| `set_parameters` | Deserialise and load weight arrays; no-ops silently on empty list |
| `fit(parameters, config)` | Local training; returns updated parameters + metrics |
| `evaluate(parameters, config)` | Local evaluation; returns loss + metrics |
| `predict(parameters, input)` | One-shot inference for a single input sample |

**Important**: `interface.py` calls `set_parameters` directly before dispatching `fit` or `evaluate`. An empty `parameters` list is legal on round 0 (the server has not yet collected global weights); the hook skips loading silently.

### FL round flow

```
1. flower-server  →  GET  /get_parameters    (from CLIENT_1_IP on startup)
2. flower-server  →  POST /evaluate          (initial global evaluation)
3. for each round:
     flower-server  →  POST /fit             (local training)
     flower-server  →  POST /evaluate        (post-training evaluation)
     flower-server  aggregates parameters via FedAvg
4. flower-server stores final global model
```

### Supported aggregation algorithms

Configured on the aggregator via `FL_ALGORITHM`:

`FedAvg` (default), `FedAvgM`, `FedProx`, `FedOpt`, `FedMedian`, `Krum`

---

## 4. HTTP API Reference

### ml-app (port 5000)

All endpoints are prefixed with their section path. During an active FL round, separation and classification endpoints return `503 Service Unavailable`.

#### Separation & playback

| Method | Path | Description |
|---|---|---|
| `GET` | `/external/separate/info` | Returns `{stem, model_name, sample_rate, busy}` |
| `POST` | `/external/separate` | Accepts a multipart audio file. Returns client's stem as WAV, or all four stems as ZIP |
| `POST` | `/external/separate/buffer` | Separates and stores the stem locally without streaming. Used for synchronised playback |
| `POST` | `/external/playback` | Plays the buffered stem via `ffplay` in the background |
| `GET` | `/external/playback/status` | Returns `{playing, stem, buffered}` |
| `DELETE` | `/external/playback` | Stops playback |
| `GET` | `/external/separate/stream` | Stream the buffered stem WAV |

#### Classification

| Method | Path | Description |
|---|---|---|
| `POST` | `/external/classify` | Classify whether the assigned stem is present in an uploaded audio file. Returns `{"stem": "vocals", "present": true, "confidence": 0.87}`. If `TARGET_STEM` is unset, returns all four stems. The dashboard pre-truncates the upload to the first 10 s before sending, keeping payloads small over a mesh link. |

#### Model management

| Method | Path | Description |
|---|---|---|
| `POST` | `/external/model/load` | Upload a `.pt` checkpoint and hot-swap it as the running model. Architecture is auto-detected from checkpoint weight shapes; model is rebuilt if `base_filters` differs. Returns `503` during FL round. |
| `GET` | `/external/model/download` | Download the current model state dict as a `.pt` file |

#### Training status

| Method | Path | Description |
|---|---|---|
| `GET` | `/training/status` | Returns `{busy, round, metrics}` — used by the dashboard training monitor |

---

### Dashboard API (port 8000)

The dashboard backend proxies commands to the registered ml-app nodes. All persistent state (uploaded files, node list, model library) is stored under `DATA_DIR` (container: `/app/data`, mounted as a volume).

#### Nodes

| Method | Path | Body / Query | Description |
|---|---|---|---|
| `GET` | `/api/nodes` | — | List registered node URLs |
| `POST` | `/api/nodes` | `{url}` | Register a node |
| `DELETE` | `/api/nodes` | `{url}` | Remove a node |
| `GET` | `/api/nodes/status` | — | Poll all nodes and return their health + playback status |

#### Audio library

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/audio` | List uploaded audio files |
| `POST` | `/api/audio/upload` | Upload a WAV / MP3 / FLAC / OGG file (max 50 MB) |
| `DELETE` | `/api/audio/{filename}` | Delete an audio file |
| `POST` | `/api/audio/buffer` | Buffer a track on a set of nodes in parallel (`{filename, nodes[]}`) |
| `POST` | `/api/audio/playback/start` | Trigger synchronised playback on a set of nodes (`{nodes[]}`) |
| `POST` | `/api/audio/playback/stop` | Stop playback on a set of nodes (`{nodes[]}`) |
| `POST` | `/api/audio/classify` | Ask a node to classify stem presence in a local file (`{filename, node_url}`) |
| `GET` | `/api/audio/stream` | Proxy-stream the buffered stem from a node (`?node_url=`) |

#### Model library

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/models` | List uploaded model checkpoints |
| `POST` | `/api/models/upload` | Upload a `.pt` checkpoint |
| `DELETE` | `/api/models/{filename}` | Delete a checkpoint |
| `POST` | `/api/models/push` | Push a checkpoint to one or more nodes (`{filename, nodes[]}`) — triggers `/external/model/load` on each node in parallel |
| `GET` | `/api/models/download` | Download the current model state dict from a node (`?node_url=`) |

#### Training monitor

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/training/status` | Poll all nodes for `{busy, round, metrics}` |

---

## 5. Dashboard

The web dashboard (`dashboard/`) is served at `http://<AGGREGATOR_IP>:8000` and provides:

- **KPI row** — total nodes, online count, playing count
- **FL Training Monitor** — polls all nodes' `/training/status` every 5 s during training; shows per-node round, loss, SI-SDR, and classification accuracy. Each node card displays a **loss sparkline** (SVG polyline of loss over fit rounds, green if falling, red if rising).
- **Audio Control Deck**
  - Upload audio files (drag-and-drop or file picker)
  - **Buffer Track** — pushes the selected track to all online nodes in parallel via a streaming NDJSON response; each node card shows a live 3-step progress indicator (Uploading → Separating → Buffered)
  - **Play All / Stop** — triggers synchronised ffplay playback on all buffered nodes and simultaneously starts in-browser audio playback with a live frequency-bar waveform visualiser on each node card
  - Per-node playback controls (individual play / stop / classify)
- **Keyboard shortcuts** — `Space` Play All / Stop, `B` Buffer Track, `R` Refresh nodes
- **Audio Library** — list, select, and delete uploaded tracks
- **Model Library** — upload a `.pt` checkpoint, then push it to any subset of online nodes to override their running weights without restarting; supports checkpoints with different `base_filters` (architecture is auto-adapted on the node)
- **FLAPS Client Nodes** — add / remove nodes by URL; live health badges; in-browser stem audio player with waveform visualiser per node
- **Orchestration Feed** — scrollable activity log with colour-coded severity levels and SVG-illustrated empty states

### Running the dashboard locally

```bash
cd dashboard
pip install -r requirements.txt
DATA_DIR=./data uvicorn app:app --reload --port 8000
```

---

## 6. Configuration Reference

### ML app environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `unet_small` | `unet_small` or `unet_large` |
| `BASE_FILTERS` | `8` | Base channel width of the U-Net |
| `SAMPLE_RATE` | `8000` | Target sample rate (Hz) |
| `N_FFT` | `512` | STFT window size |
| `HOP_LENGTH` | `128` | STFT hop |
| `CLIENT_ID` | — | Zero-based client index (required) |
| `NUM_CLIENTS` | — | Total number of FL clients (required) |
| `TARGET_STEM` | — | Stem to specialise on: `vocals`, `drums`, `bass`, `other` |
| `DATA_ROOT` | `/app/dataset` | Path to MUSDB18HQ dataset |
| `SEGMENT_SECONDS` | `2.0` | Segment window length in seconds |
| `TRAIN_CHUNK_DURATION` | `2.0` | Training chunk duration |
| `TRAIN_CHUNK_OVERLAP` | `0.0` | Overlap between training chunks |
| `MAX_TRAIN_SEGMENTS` | `10` | Max segments sampled per track for training |
| `MAX_EVAL_SEGMENTS` | `4` | Max segments sampled per track for evaluation |
| `EVAL_BATCH_SIZE` | `1` | Evaluation batch size |
| `USE_AMP` | `false` | Enable mixed-precision training (recommended `true` on Jetson) |
| `AUGMENT_TRAIN` | `false` | Enable training augmentation |
| `AUGMENT_GAIN_DB` | `6.0` | Random gain range (±dB) |
| `AUGMENT_NOISE_STD` | `0.0` | Additive noise std dev (0 = off) |
| `AUGMENT_PHASE_FLIP` | `false` | Random phase flip |
| `AUGMENT_STEM_DROPOUT_PROB` | `0.3` | Probability of zeroing a stem |
| `AUGMENT_REMIX_PROB` | `0.0` | Probability of remixing stems across tracks |
| `LOSS_CLS_WEIGHT` | `0.1` | Classification loss weight |
| `LEARNING_RATE` | `1e-3` | Initial learning rate |
| `WEIGHT_DECAY` | `1e-4` | Adam weight decay |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `cuda:0` |
| `PREDICT_CHUNK_DURATION` | `30.0` | Chunk size in seconds for inference on full tracks. Chunked inference prevents OOM on Jetson. Set to `0` to disable chunking (not recommended for long tracks). |
| `MANIFEST_CACHE_DIR` | `/tmp/musdb-manifest-cache` | Writable directory for the dataset manifest cache. Must be outside the dataset mount, which is read-only in containers. |
| `FULL_TRACKS` | `false` | If `true`, disable Non-IID partitioning |
| `LOGGING_LEVEL` | `INFO` | Python logging level |

### Aggregator (flower-server) environment variables

| Variable | Default | Description |
|---|---|---|
| `N_CLIENTS` | `2` | Expected number of FL clients |
| `N_ROUNDS` | `3` | Number of FL rounds |
| `N_EPOCHS` | `1` | Local epochs per round |
| `BATCH_SIZE` | `1` | Training batch size |
| `FL_ALGORITHM` | `FedAvg` | Aggregation algorithm |
| `ML_APP_ADDRESS` | — | URL of one ml-app (initial parameter fetch) |
| `ROUND_TIMEOUT` | `0` | Per-round timeout in seconds (0 = infinite) |
| `MAESTRO_MANAGER_ADDRESS` | `""` | Optional MobFedLS manager address |

### Dashboard environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./` | Persistent data directory (uploads, models, nodes.json) |

---

## 7. Deployment

### Physical cluster (3× Jetson)

**Node assignment:**

| Jetson | Mesh IP | Role | `CLIENT_ID` | `TARGET_STEM` |
|---|---|---|---|---|
| Jetson 638 | 192.168.100.3 | Aggregator | — | — |
| Jetson 681 | 192.168.100.2 | Client 0 | `0` | `vocals` |
| Jetson 688 | 192.168.100.1 | Client 1 | `1` | `drums` |

**Step 1 — prepare env files (on every Jetson):**

```bash
cp .env.jetson.example .env
# Edit .env to set the mesh IPs and per-node identity
```

**Step 2 — start client Jetsons first** (flower-server needs to reach a client on startup):

```bash
# Jetson 681
CLIENT_ID=0 TARGET_STEM=vocals docker compose -f docker-compose.jetson-client.yaml up --build -d

# Jetson 688
CLIENT_ID=1 TARGET_STEM=drums docker compose -f docker-compose.jetson-client.yaml up --build -d
```

Wait until `docker ps` shows both ml-apps as healthy (the healthcheck polls `/docs` every 30 s; `start_period` is 90 s).

**Step 3 — start the aggregator** (Jetson 638):

```bash
docker compose -f docker-compose.jetson-aggregator.yaml up --build
```

This starts flower-server on port 8080 and the dashboard on port 8000.

**Step 4 — trigger training:**

The flower-server begins training automatically as soon as `N_CLIENTS` ghost clients connect. Monitor progress at `http://192.168.100.3:8000`.

**Step 5 — run the demo:**

```bash
# From Jetson 638 (or any machine on the mesh)
python scripts/distribute_song.py --song mix.wav \
    --clients http://192.168.100.2:5000 http://192.168.100.1:5000
```

### Single-machine (development)

**Standalone single node:**

```bash
cd MobFedLS/tools/ml-apps/flaps
docker compose up --build
```

**Full federated stack on one machine:**

```bash
cd MobFedLS/tools/ml-apps/flaps
docker compose -f docker-compose.federated.yaml up
# Then start training:
docker compose -f docker-compose.federated.yaml --profile training up
# Or use the convenience wrapper:
./start-training.sh
```

### Model push workflow (post-training)

After training, a better checkpoint can be hot-swapped onto running nodes without any restart:

1. Open the dashboard at `http://<AGGREGATOR_IP>:8000`.
2. In **Model Library**, click **Upload .pt** and select your checkpoint.
3. Click **Push** next to the uploaded file; select the target nodes.
4. Click **Push to Selected Nodes** — the dashboard forwards the checkpoint to each node's `/external/model/load`, which auto-detects the architecture and hot-swaps the weights.

If the checkpoint was trained with a different `BASE_FILTERS` than the running node, the node rebuilds its model automatically to match.

---

## 8. Local Development

### Install dependencies

```bash
pip install -r MobFedLS/tools/ml-apps/flaps/requirements-mlapp.txt
```

### Standalone training

```bash
python MobFedLS/tools/ml-apps/flaps/scripts/train_musdb.py \
    --config scripts/train_musdb.example.json
```

### Inference

```bash
# From an audio file
python MobFedLS/tools/ml-apps/flaps/scripts/predict_from_file.py

# Real-time microphone inference
python MobFedLS/tools/ml-apps/flaps/scripts/predict_from_mic.py
```

### Synchronised demo (without Docker)

```bash
python MobFedLS/tools/ml-apps/flaps/scripts/distribute_song.py \
    --song mix.wav \
    --clients http://192.168.100.2:5000 http://192.168.100.1:5000

# Stop playback
python MobFedLS/tools/ml-apps/flaps/scripts/distribute_song.py --stop \
    --clients http://192.168.100.2:5000 http://192.168.100.1:5000

# Non-synchronised mode
python MobFedLS/tools/ml-apps/flaps/scripts/distribute_song.py \
    --song mix.wav --no-sync \
    --clients http://192.168.100.2:5000 http://192.168.100.1:5000
```

---

## 9. Test Suite

```bash
# All tests
pytest MobFedLS/tools/ml-apps/flaps/tests/

# Individual files
pytest MobFedLS/tools/ml-apps/flaps/tests/test_model_shapes.py
pytest MobFedLS/tools/ml-apps/flaps/tests/test_dataset_manifest.py
pytest MobFedLS/tools/ml-apps/flaps/tests/test_audio_utils.py
pytest MobFedLS/tools/ml-apps/flaps/tests/test_mfls_app_smoke.py
pytest MobFedLS/tools/ml-apps/flaps/tests/test_client_ml_bridge.py
pytest MobFedLS/tools/ml-apps/flaps/tests/test_file_predict.py
pytest MobFedLS/tools/ml-apps/flaps/tests/test_mic_predict.py
```

| Test file | What it covers |
|---|---|
| `test_model_shapes.py` | Forward-pass output shapes for `UNetSmall` and `UNetLarge` |
| `test_dataset_manifest.py` | Manifest generation and hash-based client partitioning |
| `test_audio_utils.py` | STFT/iSTFT round-trip fidelity and SI-SDR correctness |
| `test_mfls_app_smoke.py` | `fit` / `evaluate` / `predict` smoke tests for `MLApp` |
| `test_client_ml_bridge.py` | `clientML` hook contract (get/set/fit/evaluate) |
| `test_file_predict.py` | File-based inference pipeline |
| `test_mic_predict.py` | Microphone inference pipeline |

---

## 10. Project Structure

```
rsa-project/
├── MobFedLS/
│   ├── tools/
│   │   ├── ml-apps/flaps/              # FLAPS ML application (edit here)
│   │   │   ├── cmd/
│   │   │   │   ├── clientML.py         # MobFedLS hook entry points
│   │   │   │   ├── separate.py         # Separation / playback / model / classify endpoints
│   │   │   │   ├── main.py             # Uvicorn entrypoint
│   │   │   │   └── ml_app/
│   │   │   │       ├── mfls_app.py     # MLApp: training/eval/predict controller
│   │   │   │       ├── models/
│   │   │   │       │   ├── unet_small.py
│   │   │   │       │   └── unet_large.py
│   │   │   │       ├── dataset/
│   │   │   │       │   └── musdb_loader.py
│   │   │   │       └── utils/
│   │   │   │           ├── audio.py    # STFT/iSTFT helpers, SI-SDR
│   │   │   │           └── losses.py   # Composite separation loss
│   │   │   ├── scripts/
│   │   │   │   ├── train_musdb.py
│   │   │   │   ├── predict_from_file.py
│   │   │   │   ├── predict_from_mic.py
│   │   │   │   ├── distribute_song.py
│   │   │   │   └── decode_parameters.py
│   │   │   ├── tests/
│   │   │   ├── deployment/
│   │   │   │   └── Dockerfile          # PyTorch + FastAPI + MobFedLS; ARM64 path for Jetson
│   │   │   ├── docs/flaps/
│   │   │   │   └── training-params.md
│   │   │   ├── docker-compose.yaml                    # standalone single-node dev
│   │   │   ├── docker-compose.federated.yaml          # full stack, one machine
│   │   │   ├── docker-compose.jetson-aggregator.yaml  # Jetson 638 (flower-server + dashboard)
│   │   │   ├── docker-compose.jetson-client.yaml      # Jetson 681 / 688 (ml-app + ghost client)
│   │   │   ├── .env.jetson.example
│   │   │   ├── start-training.sh
│   │   │   └── requirements-mlapp.txt
│   │   ├── flower-ghostclient/         # gRPC↔HTTP bridge (git submodule)
│   │   └── flower-server/              # Flower FedAvg server (git submodule)
│   ├── cmd/
│   │   ├── find-neighbours/            # Mesh peer discovery (git submodule, not yet wired)
│   │   └── mfls-manager/              # Lifecycle orchestration (git submodule, not yet wired)
│   ├── internal/mfls-interface/        # HTTP hook interface contract (git submodule)
│   └── assets/
│       └── datasets/musdb18hq/         # MUSDB18HQ (mounted at /app/dataset in containers)
│
├── dashboard/
│   ├── app.py                          # FastAPI backend
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   ├── Dockerfile
│   └── requirements.txt
│
└── CLAUDE.md
```

### Submodule notes

The framework components (`flower-server`, `flower-ghostclient`, `mfls-interface`, `mfls-common`, `find-neighbours`, `mfls-manager`) are git submodules from the `ai4sme/MobFedLearnSys` upstream. Only the FLAPS application under `MobFedLS/tools/ml-apps/flaps/` and the `dashboard/` directory should be edited directly in this repo.

---

## 11. Known Limitations / To Do

### 802.11s mesh network setup

The physical network layer (IEEE 802.11s mesh point mode on Jetson Wi-Fi adapters) is not yet configured. The Jetsons currently communicate over a wired LAN at `192.168.100.x`. `find-neighbours` (peer discovery service in `MobFedLS/cmd/`) exists as a skeleton but is not wired to FLAPS.

### Maestro Manager / MobFedLS orchestration

`mfls-manager` handles node lifecycle and round orchestration in the full MobFedLS framework. `MAESTRO_MANAGER_ADDRESS` is currently unset — the server skips the end-of-training callback. Integrating the manager would enable automated round sequencing and `find-neighbours`-based topology.

### EmuCD emulation

Benchmarking FL convergence under controlled packet loss, mobility, and temporary disconnections using EmuCD has not been set up.

### Song distribution over mesh

The separation and playback endpoints are fully implemented. The central Jetson can already send a song to all clients in parallel, buffer stems, and trigger synchronised playback via `distribute_song.py`. What remains is validating this over the 802.11s mesh once the physical layer is configured.

### Dataset cache invalidation

Changing `SEGMENT_SECONDS`, `HOP_LENGTH`, `SAMPLE_RATE`, or `MONO` requires deleting the manifest cache under `MANIFEST_CACHE_DIR` (default `/tmp/musdb-manifest-cache`) or setting a different `MANIFEST_CACHE_DIR`. The `_v2` filename suffix prevents stale v1 manifests from being loaded, but there is no automatic invalidation when other parameters change.

### In-browser audio requires HTTPS on some browsers

The Web Audio API waveform visualiser in the dashboard uses `AudioContext` and `createMediaElementSource`. Some browsers block these APIs on plain HTTP origins. If the waveform does not appear, serve the dashboard over HTTPS or use a browser that permits it on `localhost`.