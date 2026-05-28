# FLAPS — Full System Architecture

Five diagrams covering the physical deployment, internal components, model pipeline, FL training round, and demo flow.

---

## 1. Physical Deployment and Communication

Three Jetson nodes on an IEEE 802.11s mesh. The aggregator runs the Flower server, two gRPC↔HTTP bridge containers, and the web dashboard. Each client runs a single ml-app container.

```mermaid
graph TB
    BROWSER(["Browser"])

    subgraph MESH["IEEE 802.11s Mesh — 192.168.100.0/24"]
        subgraph AGG["Jetson 638 — Aggregator  ·  192.168.100.3"]
            FS["flower-server\nport 8080\nFedAvg / FedAvgM / FedProx / FedOpt / Krum"]
            DASH["dashboard\nport 8000\nFastAPI + vanilla JS"]
        end
        
        subgraph C0["Jetson 681 — Client 0  ·  192.168.100.2"]
            APP0["ml-app-flaps\nport 5000  ·  FastAPI\nTARGET_STEM = vocals"]
            GC0["flower-ghostclient-0\ngRPC ↔ HTTP bridge"]
            FS -->|gRPC| GC0
        end

        subgraph C1["Jetson 688 — Client 1  ·  192.168.100.1"]
            APP1["ml-app-flaps\nport 5000  ·  FastAPI\nTARGET_STEM = drums"]
            GC1["flower-ghostclient-1\ngRPC ↔ HTTP bridge"]
            FS -->|gRPC| GC1
        end

        GC0 -->|"HTTP  /fit  /evaluate  /get_parameters"| APP0
        GC1 -->|"HTTP  /fit  /evaluate  /get_parameters"| APP1
        DASH -->|"HTTP  /external/separate/buffer  /playback  /classify  /training/status"| APP0
        DASH -->|"HTTP  /external/separate/buffer  /playback  /classify  /training/status"| APP1
    end

    BROWSER -->|"HTTP  :8000"| DASH
```

---

## 2. ML App Internal Components

Each client node runs `ml-app-flaps`, a FastAPI server whose internals are organised into five layers: API, controller, model, dataset, and utilities.

```mermaid
graph TD
    subgraph MLAPP["ml-app-flaps  —  FastAPI :5000"]
        subgraph API["API Layer"]
            FL_HOOKS["clientML.py\nget_parameters · set_parameters\nfit · evaluate · predict"]
            SEP_API["separate.py\n/external/separate · /buffer · /stream\n/external/playback  ·  /classify\n/external/model/load · /download\n/training/status"]
        end

        subgraph CTL["MLApp Controller  (mfls_app.py)"]
            LOCK["asyncio.Lock\nFL-busy guard — 503 during round"]
            LOOP["Training Loop\nAdam + ReduceLROnPlateau\nGradScaler  (USE_AMP=true on Jetson)"]
        end

        subgraph MDL["Model  (unet_small / unet_large)"]
            UNET["U-Net Encoder–Decoder\nSpectrogram domain\nbase_filters = 8  (~118 k params)\nor 24  (~1 M params)"]
            SEP_H["Separation Head\n1×1 Conv → sigmoid\nmasks  (B, 4, F, T)"]
            CLS_H["Classification Head\nBottleneck → GAP → Linear\nlogits  (B, 4)  —  no extra params"]
        end

        subgraph DS["Dataset  (musdb_loader.py)"]
            MANIFEST["Manifest v2\nstart_sample / num_samples\nstored at source sample rate (44 100 Hz)"]
            PARTITION["Non-IID Partition\nhash(track_name) % NUM_CLIENTS == CLIENT_ID"]
            AUG["Augmentation  (AUGMENT_TRAIN=true)\ngain · noise · phase-flip\nstem-dropout · stem-remix"]
        end

        subgraph UTILS["Utilities"]
            AU["audio.py\nSTFT · iSTFT\nOLA chunked inference  (Hann window)\nload_audio_segment  (seek at source SR → resample)\nSI-SDR metric"]
            LU["losses.py\nCompositeSourceSeparationLoss\nmask-BCE · magnitude-L1 · consistency\nKL regularisation · SI-SDR · cls-BCE"]
        end

        FL_HOOKS --> CTL
        SEP_API --> CTL
        CTL --> MDL
        CTL --> DS
        LOOP --> UTILS
        MDL --> AU
    end

    MUSDB[("MUSDB18HQ\n/app/dataset")] --> DS
    CKPT[("Checkpoints\n/app/logs")] <-->|best per round| LOOP
    FFPLAY(["ffplay"]) <-->|subprocess Popen| SEP_API
```

---

## 3. U-Net Model and Audio Pipeline

The full forward pass from raw waveform to separated stems and stem-presence classification.

```mermaid
graph LR
    MIX(["Audio Mix\nwaveform"])

    subgraph PREP["Audio Preparation"]
        OLA["OLA Chunker\nchunk = 30 s\nHann window\noverlap-add on output"]
        STFT_F["STFT\nn_fft = 512\nhop = 128\nsr = 8 000 Hz"]
        SPEC["Magnitude Spectrogram\nB × 1 × F × T"]
    end

    subgraph ENCODER["U-Net Encoder"]
        E1["Conv block 1\nConv2d + BN + ReLU\n↓ stride"]
        E2["Conv block 2"]
        EN["..."]
        BOTT["Bottleneck"]
    end

    subgraph DECODER["U-Net Decoder"]
        D1["Deconv block 1\nConvTranspose2d + skip"]
        D2["Deconv block 2"]
        DN["..."]
    end

    subgraph HEADS["Output Heads"]
        SEP["Separation Head\n1×1 Conv → sigmoid\nmasks  B × 4 × F × T"]
        CLS["Classification Head\nGAP → Linear\nlogits  B × 4"]
    end

    subgraph OUT["Output"]
        MASKED["Apply masks to mix spectrogram"]
        ISTFT_F["iSTFT + OLA accumulate"]
        STEMS(["4 separated stems\nvocals · drums · bass · other"])
        PRESENCE(["Stem presence\n{stem, present, confidence}"])
    end

    MIX --> OLA --> STFT_F --> SPEC
    SPEC --> E1 --> E2 --> EN --> BOTT
    BOTT --> D1 --> D2 --> DN --> SEP
    BOTT --> CLS
    SEP --> MASKED
    SPEC -->|mix| MASKED
    MASKED --> ISTFT_F --> STEMS
    CLS --> PRESENCE
```

---

## 4. Federated Learning Round

One complete FL cycle: initial parameter fetch, per-round local training, FedAvg aggregation, and post-round evaluation.

```mermaid
sequenceDiagram
    participant FS as flower-server
    participant GC0 as ghostclient-0
    participant GC1 as ghostclient-1
    participant A0 as ml-app-0  vocals
    participant A1 as ml-app-1  drums

    Note over FS,A1: Startup — initial parameter fetch
    FS->>GC0: GET /get_parameters
    GC0->>A0: HTTP GET /get_parameters
    A0-->>GC0: flat NumPy weight arrays
    GC0-->>FS: initial global weights

    Note over FS,A1: Round 0 — baseline evaluation
    FS->>GC0: POST /evaluate (global weights)
    GC0->>A0: HTTP POST /evaluate
    A0-->>GC0: loss · SI-SDR · cls_accuracy
    FS->>GC1: POST /evaluate (global weights)
    GC1->>A1: HTTP POST /evaluate
    A1-->>GC1: loss · SI-SDR · cls_accuracy

    loop Each FL round  (N_ROUNDS = 3)
        Note over FS,A1: Local training — runs in parallel on both clients
        FS->>GC0: POST /fit (weights · epochs · batch_size)
        GC0->>A0: HTTP POST /fit
        A0->>A0: train on vocals partition (Non-IID)<br/>loss active only on TARGET_STEM channel
        A0-->>GC0: updated weights + metrics
        FS->>GC1: POST /fit (weights · epochs · batch_size)
        GC1->>A1: HTTP POST /fit
        A1->>A1: train on drums partition (Non-IID)<br/>loss active only on TARGET_STEM channel
        A1-->>GC1: updated weights + metrics

        Note over FS: FedAvg aggregation
        FS->>FS: w_global = weighted_avg(w0, w1)<br/>weight = num_examples per client

        Note over FS,A1: Post-round evaluation with new global model
        FS->>GC0: POST /evaluate (w_global)
        GC0->>A0: HTTP POST /evaluate
        A0-->>GC0: metrics
        FS->>GC1: POST /evaluate (w_global)
        GC1->>A1: HTTP POST /evaluate
        A1-->>GC1: metrics
    end

    Note over FS: Training complete — final global model stored
```

---

## 5. Demo Flow: Song Distribution and Synchronised Playback

Dashboard-driven flow from audio upload to distributed stem playback, including optional stem-presence classification.

```mermaid
sequenceDiagram
    participant BR as Browser
    participant DA as Dashboard :8000
    participant A0 as ml-app-0 :5000  vocals
    participant A1 as ml-app-1 :5000  drums
    participant FF as ffplay  (per node)

    Note over BR,FF: Buffer Track — distribute song to both nodes in parallel
    BR->>DA: POST /api/audio/buffer  {filename, nodes[]}
    DA-->>BR: NDJSON stream — progress per node

    par node 0
        DA->>A0: POST /external/separate/buffer  (audio file)
        Note over DA,A0: dashboard NDJSON: step = uploading
        A0->>A0: U-Net chunked inference → vocals stem
        Note over DA,A0: dashboard NDJSON: step = separating
        A0-->>DA: {ready: true, stem: "vocals"}
        Note over DA,A0: dashboard NDJSON: step = buffered
    and node 1
        DA->>A1: POST /external/separate/buffer  (audio file)
        A1->>A1: U-Net chunked inference → drums stem
        A1-->>DA: {ready: true, stem: "drums"}
    end

    Note over BR,FF: Classify (optional — uses first 10 s snippet)
    BR->>DA: POST /api/audio/classify  {filename, node_url}
    DA->>A0: POST /external/classify  (10 s WAV snippet)
    A0->>A0: classification head forward pass
    A0-->>DA: {stem: "vocals", present: true, confidence: 0.87}
    DA-->>BR: classification result

    Note over BR,FF: Play All — synchronised ffplay + in-browser audio
    BR->>DA: POST /api/audio/playback/start  {nodes[]}

    par node 0
        DA->>A0: POST /external/playback
        A0->>FF: subprocess Popen  ffplay vocals.wav
        FF-->>A0: playing (non-blocking)
    and node 1
        DA->>A1: POST /external/playback
        A1->>FF: subprocess Popen  ffplay drums.wav
        FF-->>A1: playing (non-blocking)
    end

    DA-->>BR: {playing: true}
    Note over BR: GET /api/audio/stream → in-browser audio element<br/>Web Audio API waveform visualiser active per node card

    Note over BR,FF: Stop
    BR->>DA: POST /api/audio/playback/stop  {nodes[]}
    par
        DA->>A0: DELETE /external/playback
        A0->>FF: terminate()
    and
        DA->>A1: DELETE /external/playback
        A1->>FF: terminate()
    end
    DA-->>BR: {stopped: true}
```