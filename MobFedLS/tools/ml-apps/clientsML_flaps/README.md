# FLAPS: Federated Learning Acoustic Perception and Separation

This ML-App implements a multi-task model for both **Acoustic Perception** (classification of instruments) and **Source Separation** (isolating instruments from a mix) using Federated Learning.

## Features
- **Shared Encoder**: Extracts common acoustic features.
- **Perception Head**: Multi-label classification to identify the presence of Drums, Bass, Vocals, and Other instruments.
- **Separation Heads**: 4 distinct decoder heads for separating each instrument.
- **Federated Specialization**: Each node can be assigned an `INSTRUMENT_PROFILE` (e.g., `drums`). It will specialize in training that separation head while contributing to the global perception model.

## Prerequisites
- MUSDB18-HQ dataset (mounted at `/app/dataset`)
- Docker and MobFedLS framework

## Setup

1. **Build the Image**
   ```bash
   docker build -t ml-app-flaps -f MobFedLS/tools/ml-apps/clientsML_flaps/Dockerfile.clientML_AND_interface MobFedLS/tools/ml-apps/clientsML_flaps/
   ```

2. **Environment Variables**
   - `INSTRUMENT_PROFILE`: The instrument this node specializes in (`drums`, `bass`, `vocals`, `other`).
   - `DATASET`: (Optional) Name of the config file.
   - `LOGGING_LEVEL`: `DEBUG`, `INFO`, etc.

3. **Docker Compose**
   Add the following to your `docker-compose.yaml`:
   ```yaml
   ml-app:
     image: ml-app-flaps
     environment:
       - LOGGING_LEVEL=INFO
       - INSTRUMENT_PROFILE=drums # Change per node
       - PLOT_PERFORMANCE=True
     volumes:
       - './musdb18hq:/app/dataset'
       - './logs:/app/logs'
     ports:
       - "5001:5000"
   ```

## Usage
1. Trigger `get_data` to load a subset of MUSDB18-HQ.
2. Trigger `fit` to start local training.
3. Use `start_aggregation` from the server node to begin the Federated Learning process.
4. After aggregation, use `predict` to see the separation results in the logs.

## Dataset Structure
The app expects the MUSDB18-HQ structure at `/app/dataset`:
```
/app/dataset/
├── train/
│   ├── Song Name/
│   │   ├── mixture.wav
│   │   ├── drums.wav
│   │   ├── bass.wav
│   │   └── ...
└── test/
    └── ...
```
