# FLAPS ML App

This folder contains the PyTorch-based `FLAPS` ML app used by the MobFedLS interface layer.

## What it provides
- `clientML.py`: the MobFedLS hook module required by `internal/mfls-interface`.
- `ml_app/`: the actual model, dataset, loss, and audio processing code.
- `deployment/`: a container image that bundles the app with the MobFedLS interface base image.
- `scripts/`: local training and inference helpers.

## Required runtime hooks
The MobFedLS interface imports the following functions from `clientML.py`:
- `get_parameters()`
- `set_parameters(parameters)`
- `get_data()`
- `fit(config)`
- `evaluate(config)`
- `predict(plot_graphs, before_after)`
- `in_usage(value)`
- `set_run_path(run_path)`

## Build
From the repo root:

```bash
docker build -t ml-app-flaps -f MobFedLS/tools/ml-apps/flaps/deployment/Dockerfile MobFedLS/tools/ml-apps/flaps/
```

On Jetson / `arm64`, the Dockerfile installs `torch` from:
`https://pypi.jetson-ai-lab.io/jp6/cu126`

If you're cross-building from an x86 machine, prefer building natively on the Jetson or with a true `linux/arm64` build context so pip resolves the matching arm64 wheels.

## Docker Compose
To start the app locally with the included Compose file:

```bash
cd MobFedLS/tools/ml-apps/flaps
docker compose up --build
```

The default Compose setup mounts:
- `MobFedLS/assets/datasets/musdb18hq` into `/app/dataset`
- `MobFedLS/assets/logs` into `/app/logs`
- `/var/run/docker.sock` into the container so the MobFedLS interface can resolve container metadata

## Useful environment variables
- `DATA_ROOT` / `MUSDB_ROOT`: path to the MUSDB18HQ root.
- `FULL_TRACKS=true`: switch the app to full-track loading.
- `TRAIN_CHUNK_DURATION`: chunk length in seconds for memory-safe training.
- `TRAIN_CHUNK_OVERLAP`: chunk overlap ratio.
- `MODEL_NAME`: `unet_small` or `unet_large`.
- `BASE_FILTERS`: width of the U-Net.
- `SAMPLE_RATE`: target sample rate.
- `USE_AMP=true`: enable mixed precision on CUDA.
- `EVAL_BATCH_SIZE`: batch size used for evaluation.

## MobFedLS integration
The app is driven by the standard MobFedLS flow:
1. `internal/mfls-interface` calls `get_data()` and `get_parameters()`.
2. Flower ghost clients proxy `fit`/`evaluate` requests through the interface.
3. The Flower server aggregates the returned parameters with the chosen strategy (for example, `FedAvg`).

If you want to run this app in full-track federated mode, set `FULL_TRACKS=true` and a small `TRAIN_CHUNK_DURATION` such as `2.5` or `3.0` seconds.

