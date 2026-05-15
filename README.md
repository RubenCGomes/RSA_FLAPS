 # RSA_FLAPS: Federated Learning Acoustic Perception and Separation

This project implements a decentralized acoustic intelligence system and a distributed source separation pipeline using **Federated Learning (FL)**. Nodes collaborate to recognize environmental sounds and isolate individual instruments from shared audio mixes without centralizing raw data.

## What has been done

### 1. Project Foundation & Architecture
- **Specification**: Defined the project proposal (`rsa-proj-flaps.md`) covering the dual use cases: **Distributed Acoustic Perception** and **Federated Source Separation**.
- **Workspace Reorganization**: Consolidated the project into the `MobFedLS` framework structure for better integration with the federated learning ecosystem.
- **Documentation**: Centralized technical guides, including CUDA memory fixes and project specifications, in `MobFedLS/docs/flaps/`.

### 2. Core Machine Learning (Local)
- **Model Architectures**: Implemented `UNetSmall` and `UNetLarge` architectures for audio source separation (PyTorch).
- **Data Pipeline**: 
    - Created a robust `MusdbSeparationDataset` loader for the **MUSDB18HQ** dataset.
    - Implemented audio preprocessing (STFT, Mel-Spectrograms) and post-processing (iSTFT).
- **Training Infrastructure**: Developed `mfls_app.py` as a centralized controller for local training, validation, and testing.
- **Evaluation**: Integrated metrics like **SI-SDR** (Scale-Invariant Source-to-Distortion Ratio) for separation quality assessment.

### 3. Tooling & Testing
- **Local Scripts**: 
    - `train_musdb.py`: For training models on local datasets.
    - `predict_from_file.py` & `predict_from_mic.py`: For real-time and file-based inference.
- **Validation Suite**: A comprehensive test suite in `MobFedLS/tools/ml-apps/flaps/tests/` covering:
    - Model input/output shapes.
    - Dataset manifest generation.
    - Audio utility correctness.
    - Smoke tests for the main application.

---

## What's still to be done (Next Steps)

### 1. Federated Learning Integration (High Priority)
- **Flower (flwr) Integration**: 
    - Implement the `FlowerClient` to wrap the local `MLApp` logic.
    - Configure the `FlowerServer` (Aggregator) for **FedAvg** updates.
- **MobFedLS Orchestration**: Register the `flaps` app within the MobFedLS framework to handle node lifecycle and discovery.

### 2. Network & Distribution Layer
- **802.11s Mesh Setup**: Configure the Raspberry Pi and Jetson nodes for peer-to-peer communication via an IEEE 802.11s self-healing mesh.
- **IPFS Distribution**: Implement the logic to publish/fetch model checkpoints and audio metadata via **IPFS** (InterPlanetary File System) for resilient content delivery.

### 3. Simulation & Real-World Deployment
- **EmuCD Emulation**: Set up mobility scenarios to benchmark FL convergence under packet loss and dynamic network conditions.
- **Physical Cluster Deployment**: Deploy the system on the **NVIDIA Jetson** (Aggregator) and **Raspberry Pi 5** (Clients) cluster.
- **Distributed Playback**: Implement the synchronized playback logic across nodes after successful source separation.

### 4. Feature Expansion
- **Acoustic Perception**: Extend the current models to explicitly handle environmental sound recognition (event detection) alongside instrument separation.
- **Refinement**: Optimize the `UNet` models for edge execution on Raspberry Pi hardware (e.g., Quantization, Pruning).

---

## Project Structure (Reorganized)
- `MobFedLS/tools/ml-apps/flaps/ml_app/`: Core PyTorch models and data logic.
- `MobFedLS/tools/ml-apps/flaps/scripts/`: Execution scripts for training and inference.
- `MobFedLS/tools/ml-apps/flaps/tests/`: Automated test suite.
- `MobFedLS/assets/datasets/musdb18hq/`: Local dataset storage.
- `MobFedLS/docs/flaps/`: Project specifications and technical documentation.
