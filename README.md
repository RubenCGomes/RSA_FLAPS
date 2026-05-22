# RSA_FLAPS: Federated Learning Acoustic Perception and Separation

This project implements a decentralized acoustic intelligence system and a distributed source separation pipeline using **Federated Learning (FL)**. Nodes collaborate to isolate individual instruments from shared audio mixes without centralizing raw data.

---

## Status

### Done

#### 1. Machine Learning Core
- **Model architectures**: `UNetSmall` and `UNetLarge` — spectrogram-domain U-Nets for 4-stem source separation (vocals, drums, bass, other), with a **dual head**: a separation mask head and a classification head (bottleneck → GAP → Linear) for stem presence detection.
- **Data pipeline**: `MusdbSeparationDataset` with deterministic hash-based **Non-IID partitioning** across clients (`hash(track_name) % NUM_CLIENTS == CLIENT_ID`), segment windowing, and augmentation (gain, noise, phase flip, stem dropout, remix).
- **Training loop**: `mfls_app.py` owns the full fit/evaluate/predict cycle with LR scheduling (ReduceLROnPlateau), AMP support, and checkpoint saving.
- **Composite loss**: mask BCE + magnitude L1 + mixture consistency + KL regularisation + SI-SDR + per-stem BCE classification loss (weighted by `LOSS_CLS_WEIGHT`).
- **Evaluation metrics**: per-stem and mean SI-SDR and SI-SDR improvement, spectral losses, classification accuracy.

#### 2. Acoustic Perception (Stem Presence Classification)
Each client trains a classification head alongside the separation masks. The head learns to detect whether its assigned stem is present in the mix:
- Presence labels are derived from spectrogram energy (`target_mag.mean() > threshold`); stems zeroed by augmentation are labelled absent automatically.
- During training, a per-stem BCE loss (`stem_classification_loss`) is added to the composite separation loss, gated on `TARGET_STEM` so each client only trains its own logit.
- `cls_loss` and `cls_accuracy` are reported per FL round in fit/evaluate metrics.

**`POST /external/classify`** — classify whether the assigned stem is present in an uploaded audio file:
```
{"stem": "vocals", "present": true, "confidence": 0.87}
```
If `TARGET_STEM` is not set, all four stems are returned.

The classification head shares the U-Net encoder and bottleneck with the separation head, so it adds no extra parameters to the aggregated model — FedAvg remains fully compatible.

#### 3. One-Class-Per-Client (Non-IID) FL Setup
- Each client is assigned a **target stem** via `TARGET_STEM` env var (e.g. client-1 → `vocals`, client-2 → `drums`).
- During training, loss is computed only on the assigned stem channel for both separation and classification; the model still outputs all 4 stems so FedAvg aggregation is compatible across clients.
- After aggregation the global model benefits from each specialisation.

#### 4. Federated Learning Integration (Flower + MobFedLS)
- **`clientML.py`**: implements the MobFedLS hook contract (`get_parameters`, `set_parameters`, `fit`, `evaluate`, `predict`).
- **`flower-ghostclient`**: gRPC↔HTTP bridge that translates the Flower protocol into HTTP calls to the ml-app FastAPI server.
- **`flower-server`**: FedAvg aggregation server (also supports FedAvgM, FedProx, FedOpt, FedMedian, Krum, and others via `FL_ALGORITHM`).
- Full federated round is operational: server fetches initial parameters from ml-app-1, ghost clients call fit/evaluate over HTTP each round, server aggregates and redistributes.

#### 5. Song Separation, Playback, and Classification
Each ml-app exposes HTTP endpoints for post-FL song distribution and synchronized playback:

| Method   | Path                        | Description                                                                                                                                                                                               |
|----------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GET`    | `/external/separate/info`   | Returns the client's assigned stem, model name, sample rate, and FL busy state.                                                                                                                           |
| `POST`   | `/external/separate`        | Accepts a multipart audio file. Returns the client's stem as `WAV`, or all four stems as `ZIP`. Responds `503` during an FL round.                                                                        |
| `POST`   | `/external/separate/buffer` | Separates and stores the stem locally without streaming it back. Used for synchronized playback — the central node calls this on all clients in parallel, then fires `/external/playback` on all at once. |
| `POST`   | `/external/playback`        | Plays the buffered stem via `ffplay` in the background. Returns `409` if already playing, `404` if nothing is buffered.                                                                                   |
| `GET`    | `/external/playback/status` | Returns `{playing, stem, buffered}`.                                                                                                                                                                      |
| `DELETE` | `/external/playback`        | Stops playback.                                                                                                                                                                                           |
| `POST`   | `/external/classify`        | Classifies whether this client's assigned stem is present in the uploaded audio. Returns `{"stem": "vocals", "present": true, "confidence": 0.87}`. Responds `503` during an FL round.                    |

**Synchronized demo flow (via `distribute_song.py`):**
```
1. GET  /external/separate/info    →  discover each client's target stem
2. POST /external/separate/buffer  →  all clients separate in parallel, buffer their stem
3. POST /external/playback         →  all clients start playing simultaneously
```

**`scripts/distribute_song.py`** — central node orchestration script:
```bash
# Run the full synchronized demo
python scripts/distribute_song.py --song mix.wav \
    --clients http://192.168.1.10:5000 http://192.168.1.11:5000

# Stop playback on all nodes
python scripts/distribute_song.py --stop \
    --clients http://192.168.1.10:5000 http://192.168.1.11:5000

# Non-synchronized mode (each node plays as soon as it finishes separating)
python scripts/distribute_song.py --song mix.wav --no-sync \
    --clients http://192.168.1.10:5000 http://192.168.1.11:5000
```

#### 6. Containerised Deployment

**Single-machine (development):**
- `docker-compose.yaml` — standalone single-node mode.
- `docker-compose.federated.yaml` — full federated stack on one machine: 2 ml-apps + flower-server + 2 ghost clients.
- `start-training.sh` — convenience wrapper for the federated run.

**Physical cluster (3x Jetson):**
- `docker-compose.jetson-aggregator.yaml` — Jetson 1: runs flower-server only.
- `docker-compose.jetson-client.yaml` — Jetson 2 & 3: runs ml-app + ghost client. Shared template parameterised by `CLIENT_ID` and `TARGET_STEM`.
- `.env.jetson.example` — copy to `.env` on each Jetson and fill in mesh IPs and per-node identity.

**Jetson deployment procedure:**
```bash
# 0. On all Jetsons: copy and edit the env file
cp .env.jetson.example .env

# 1. Start client Jetsons first (flower-server needs to reach one on startup)
#    On Jetson 2:
CLIENT_ID=0 TARGET_STEM=vocals docker compose -f docker-compose.jetson-client.yaml up --build -d

#    On Jetson 3:
CLIENT_ID=1 TARGET_STEM=drums docker compose -f docker-compose.jetson-client.yaml up --build -d

# 2. Once both client ml-apps are healthy, start the aggregator
#    On Jetson 1:
docker compose -f docker-compose.jetson-aggregator.yaml up --build

# 3. After FL training finishes, run the demo from Jetson 1:
python scripts/distribute_song.py --song mix.wav \
    --clients http://192.168.1.11:5000 http://192.168.1.12:5000
```

- `deployment/Dockerfile` — ml-app image (PyTorch + FastAPI + MobFedLS interface). Includes an ARM64 path for Jetson (`--platform linux/arm64` + Jetson PyTorch wheel).

#### 7. Local Scripts and Tooling
- `scripts/train_musdb.py` — standalone local training with JSON config.
- `scripts/predict_from_file.py` — offline inference from an audio file.
- `scripts/predict_from_mic.py` — real-time inference from a microphone.
- `scripts/decode_parameters.py` — utility to inspect serialised model weights.
- `scripts/distribute_song.py` — central node orchestration: discovers clients, distributes a song, triggers synchronized playback.

#### 8. Test Suite
- `tests/test_model_shapes.py` — forward-pass shape checks for both U-Net variants.
- `tests/test_dataset_manifest.py` — manifest generation and hash partitioning.
- `tests/test_audio_utils.py` — STFT/iSTFT round-trip and SI-SDR correctness.
- `tests/test_mfls_app_smoke.py` — smoke tests for fit/evaluate/predict calls.
- `tests/test_client_ml_bridge.py` — clientML hook contract tests.
- `tests/test_file_predict.py`, `test_mic_predict.py` — inference pipeline tests.

---

### Still To Do

#### 1. 802.11s Mesh Network Setup
The physical network layer (IEEE 802.11s mesh point mode on Jetson Wi-Fi adapters) is not yet configured. `find-neighbours` (peer discovery service in `MobFedLS/cmd/`) exists as a skeleton but is not wired to FLAPS.

#### 2. Song Distribution and Stem Playback
The separation and playback endpoints are fully implemented and tested. The central Jetson can already send a song to all clients in parallel, buffer their stems, and trigger synchronized playback via `distribute_song.py`. What remains is wiring this over the 802.11s mesh using real node IPs once the physical cluster is up.

#### 3. Maestro Manager / MobFedLS Orchestration
`mfls-manager` handles node lifecycle and round orchestration in the full MobFedLS framework. `MAESTRO_MANAGER_ADDRESS` is currently unset — the server skips the end-of-training callback. Integrating the manager would enable automated round sequencing and `find-neighbours`-based topology.

#### 4. EmuCD Emulation
Benchmarking FL convergence under controlled packet loss, mobility, and temporary disconnections using **EmuCD** has not been set up.

#### 5. Physical Cluster Deployment
Per-node docker-compose files and env templates are ready. Deployment is blocked only on the 802.11s mesh being up. Once it is, the procedure is documented in the Containerised Deployment section above.

---

## Project Structure

```
MobFedLS/
├── tools/
│   ├── ml-apps/flaps/          # FLAPS ML application (edit here)
│   │   ├── cmd/
│   │   │   ├── clientML.py         # MobFedLS hook entry points
│   │   │   ├── separate.py         # Separation + playback + classify endpoints
│   │   │   ├── main.py             # Uvicorn entrypoint (overrides base image)
│   │   │   └── ml_app/
│   │   │       ├── mfls_app.py     # Training/eval/predict controller
│   │   │       ├── models/         # UNetSmall, UNetLarge
│   │   │       ├── dataset/        # MusdbSeparationDataset
│   │   │       └── utils/          # STFT helpers, losses, SI-SDR, mic
│   │   ├── scripts/                # train, predict_from_file, predict_from_mic, distribute_song
│   │   ├── tests/                  # pytest suite
│   │   ├── deployment/Dockerfile
│   │   ├── docker-compose.yaml                   # single-node dev
│   │   ├── docker-compose.federated.yaml         # full stack, one machine
│   │   ├── docker-compose.jetson-aggregator.yaml # Jetson 1 (flower-server)
│   │   ├── docker-compose.jetson-client.yaml     # Jetson 2 & 3 (ml-app + ghost client)
│   │   ├── .env.jetson.example                   # mesh IP + per-node config template
│   │   └── start-training.sh
│   ├── flower-ghostclient/     # gRPC↔HTTP bridge (Flower client)
│   └── flower-server/          # Flower FedAvg aggregation server
├── cmd/
│   ├── find-neighbours/        # Mesh peer discovery (not yet wired)
│   └── mfls-manager/           # Lifecycle orchestration (not yet wired)
├── internal/mfls-interface/    # HTTP hook interface contract
└── assets/
    └── datasets/musdb18hq/     # MUSDB18HQ dataset (mounted at /app/dataset in containers)
```