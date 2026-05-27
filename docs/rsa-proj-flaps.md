# Project Report: Federated Learning Acoustic Perception and Separation

**Course:** Redes e Sistemas Autónomos (RSA)
**Academic Year:** 2025/2026

---

### 1. Group Members
* **Student 1:** Rúben Gomes / 113435
* **Student 2:** Simão Almeida / 113085

### 2. Description

This project implements a decentralised music source separation system where autonomous edge nodes collaborate via Federated Learning to isolate individual instrument stems from a shared audio mix. Using the MUSDB18HQ dataset as a proxy for heterogeneous, distributed real-world audio data (analogous to, e.g., distributed sensor nodes capturing environmental sound), the system demonstrates how a mesh network can learn collectively without centralising raw audio.

Two integrated capabilities are provided by each node:

1. **Federated Source Separation** — isolating each node's designated instrument stem from a full mix using a spectrogram-domain U-Net trained collaboratively via FedAvg
2. **Stem Presence Classification** — detecting whether the node's assigned stem is present in an uploaded audio clip, via a lightweight classification head that shares the separation encoder at no extra parameter cost

The network operates as a distributed "Intelligent Orchestra" where each node holds private, specialised, **Non-IID** audio data (a vocals partition or a drums partition) and shares only model weight updates.

* **The Problem:** In autonomous systems, audio data is distributed, heterogeneous, and costly to centralise. Each node observes only part of the acoustic world.
* **The FL Solution:** Using **MobFedLS** with **Flower**, nodes train locally and share only **model updates** (weights), preserving privacy and reducing bandwidth usage over the mesh.
* **Autonomous Aggregation:** Updates travel through an **IEEE 802.11s mesh network** to an **NVIDIA Jetson** aggregator that runs **FedAvg** and redistributes the improved global model.
* **Unified Capability:** After aggregation, each node can both (i) classify stem presence in an uploaded clip and (ii) extract its designated stem from a shared full mix for synchronised playback.
* **Orchestration:** Audio assets and model checkpoints are managed through a web dashboard on the aggregator, which distributes songs and model files to client nodes via HTTP over the mesh and triggers synchronised playback.

### 3. Simulation/Emulation vs. Real-World Implementation

#### Simulation/Emulation

EmuCD-based benchmarking (FL convergence under controlled packet loss and disconnections) was not completed within the project timeline.

#### Implementation

The system was developed and validated using Docker Compose on a single machine, then deployed to a physical cluster: two NVIDIA Jetson client nodes and one NVIDIA Jetson aggregator, communicating over an **IEEE 802.11s mesh network**.

### 4. Hardware Used

* **3× NVIDIA Jetson:** One as the **FL Aggregator** (Jetson 638, `192.168.100.3`), two as **FL Clients** (Jetson 681, `192.168.100.2`; Jetson 688, `192.168.100.1`).
* **3× Wi-Fi Adapters:** Supporting **802.11s mesh point mode** for peer-to-peer communication.
* **Speakers:** Used to demonstrate distributed stem playback after separation (via `ffplay` on each node).

### 5. Implementation Diagrams

The diagram below illustrates the deployed FL pipeline, from local training on private audio partitions to synchronised distributed playback over the 802.11s mesh.

```mermaid
graph TD
    subgraph "Jetson 638 — Aggregator (192.168.100.3)"
        A["Global U-Net (Separation + Classification)"] --> B{FedAvg — flower-server}
        B --> A
        D[Dashboard :8000] -- "Distribute audio / push model" --> C
        B -- "Broadcast updated weights via ghostclient" --> C
    end

    subgraph "Mesh Network (IEEE 802.11s)"
        C((802.11s Mesh))
    end

    subgraph "Jetson 681 — Client 0 (192.168.100.2)"
        D1[MUSDB18HQ — vocals partition] --> E1[Local Training]
        E1 -- "Weights via flower-ghostclient" --> C
        C -- "Global Model" --> G1[Separation + Classification]
        G1 --> H1((Buffered vocals stem → ffplay))
    end

    subgraph "Jetson 688 — Client 1 (192.168.100.1)"
        D2[MUSDB18HQ — drums partition] --> E2[Local Training]
        E2 -- "Weights via flower-ghostclient" --> C
        C -- "Global Model" --> G2[Separation + Classification]
        G2 --> H2((Buffered drums stem → ffplay))
    end

    C -- "Weight Updates" --> B
```

The sequence diagram below shows a complete end-to-end cycle: federated training, song distribution, and synchronised playback.

```mermaid
sequenceDiagram
    participant B as Browser (Dashboard)
    participant J as Jetson 638 (flower-server + dashboard)
    participant G as flower-ghostclient (×2)
    participant P0 as Jetson 681 — vocals
    participant P1 as Jetson 688 — drums

    Note over J,P1: Stage 1 — Federated Training
    J->>G: Start round (N_ROUNDS=3, N_EPOCHS=1)
    G->>P0: POST /fit {epochs, batch_size}
    P0->>P0: Train on vocals partition (Non-IID)
    P0-->>G: weights + metrics (loss, SI-SDR)
    G->>P1: POST /fit {epochs, batch_size}
    P1->>P1: Train on drums partition (Non-IID)
    P1-->>G: weights + metrics
    G-->>J: FedAvg(weights-0, weights-1) → global model

    Note over J,P1: Stage 2 — Song Distribution
    B->>J: POST /api/audio/buffer {filename, nodes}
    J->>P0: POST /external/separate/buffer (audio)
    P0->>P0: U-Net inference → buffer vocals stem
    P0-->>J: {ready: true, stem: "vocals"}
    J->>P1: POST /external/separate/buffer (audio)
    P1->>P1: U-Net inference → buffer drums stem
    P1-->>J: {ready: true, stem: "drums"}
    J-->>B: NDJSON progress stream (per-node steps)

    Note over J,P1: Stage 3 — Synchronised Playback
    B->>J: POST /api/audio/playback/start
    J->>P0: POST /external/playback
    J->>P1: POST /external/playback
    P0->>P0: ffplay vocals.wav
    P1->>P1: ffplay drums.wav
    B->>B: In-browser waveform visualiser (Web Audio API)
```