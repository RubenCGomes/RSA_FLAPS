# FLAPS Training Parameters

All parameters are set via environment variables. Values shown are the defaults used in `docker-compose.federated.yaml`.

---

## Federated Learning (Flower Server)

| Variable        | Default  | Description                                                                                                                                                                                                               |
|-----------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `N_CLIENTS`     | `2`      | Number of ghost clients that must be connected before the server starts each round. Must match the number of running `flower-client` containers.                                                                          |
| `N_ROUNDS`      | `3`      | Total number of federated rounds. Each round: server dispatches fit to all clients → collects results → aggregates weights → dispatches evaluate.                                                                         |
| `N_EPOCHS`      | `1`      | Local training epochs each client runs per round before returning updated weights. Higher values mean more local work before aggregation, which can speed up convergence but increase client drift.                       |
| `BATCH_SIZE`    | `1`      | Mini-batch size used during the local training loop. With the current CPU-only setup, 1 is the most memory-efficient choice.                                                                                              |
| `ROUND_TIMEOUT` | `0`      | Seconds the server waits for a client to return fit/evaluate results. `0` disables the timeout entirely — recommended for CPU training where a round can take several minutes.                                            |
| `FL_ALGORITHM`  | `FedAvg` | Aggregation strategy. Options: `FedAvg`, `FedAvgM`, `FedMedian`, `FedProx`, `FedOpt`, `FedAdagrad`, `FedAdam`, `FedYogi`, `FedTrimmedAvg`, `Krum`, `QFedAvg`, `FaultTolerantFedAvg`, `DPFedAvgAdaptive`, `DPFedAvgFixed`. |

---

## Dataset

| Variable          | Default               | Description                                                                                                                                                                                            |
|-------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DATA_ROOT`       | `/app/dataset`        | Path inside the container where the MUSDB18HQ dataset is mounted.                                                                                                                                      |
| `SPLIT`           | `train`               | Dataset split used for training. MUSDB18HQ provides `train` and `test`. The `val` split is carved out of `train` using a deterministic track hash (see `EVAL_SPLIT`).                                  |
| `EVAL_SPLIT`      | `val`                 | Split used during evaluation. `val` uses ~15 % of the training tracks (determined by `eval_fraction=0.15` inside the dataset loader).                                                                  |
| `FULL_TRACKS`     | `false`               | When `true`, each dataset record is a full track (no segmenting). When `false`, tracks are cut into fixed-length segments of `SEGMENT_SECONDS`. Full-track mode uses much more RAM and is much slower. |
| `SEGMENT_SECONDS` | `2.0`                 | Length of each audio segment in seconds. Ignored when `FULL_TRACKS=true`. Shorter segments mean smaller spectrograms and faster batches, at the cost of less temporal context per sample.              |
| `CLIENT_ID`       | _(set per container)_ | Zero-based index of this client. Used to partition tracks: client `k` receives tracks where `hash(track_name) % NUM_CLIENTS == k`. Each client trains on a disjoint subset of the dataset.             |
| `NUM_CLIENTS`     | `2`                   | Total number of clients. Determines the partition granularity together with `CLIENT_ID`.                                                                                                               |

---

## Audio / Spectrogram

| Variable      | Default | Description                                                                                                                                                                                                                       |
|---------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SAMPLE_RATE` | `8000`  | Target sample rate in Hz. All audio is resampled to this rate on load. Lower values reduce spectrogram size and model input dimension. 8 kHz captures speech and most musical content; 16 kHz or 44100 Hz preserves full quality. |
| `N_FFT`       | `512`   | FFT window size in samples. Determines frequency resolution: `N_FFT/2 + 1` frequency bins. Larger values give finer frequency resolution but bigger spectrograms and slower training. Should be a power of 2.                     |
| `HOP_LENGTH`  | `128`   | Number of samples between consecutive STFT frames. Controls time resolution: smaller hop = more frames per second = larger spectrogram. Typically `N_FFT/4`.                                                                      |

---

## Model Architecture

| Variable       | Default      | Description                                                                                                                                                                                                                          |
|----------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MODEL_NAME`   | `unet_small` | Model variant. `unet_small` is a lightweight U-Net suited for CPU training. `unet_large` has more layers and filters — requires significantly more memory and compute.                                                               |
| `BASE_FILTERS` | `8`          | Number of filters in the first convolutional layer of the U-Net. All subsequent layers scale from this value. Doubling `BASE_FILTERS` roughly quadruples parameter count. `8` → ~few thousand params; `32` → several million params. |
| `DEVICE`       | `auto`       | Compute device: `cpu`, `cuda`, `cuda:0`, etc. When set to `auto` or unset, the app picks CUDA if available, otherwise CPU.                                                                                                           |

---

## Training Loop

| Variable                 | Default | Description                                                                                                                                                                                                      |
|--------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MAX_TRAIN_SEGMENTS`     | `10`    | Maximum number of dataset segments used per training round. Limits how long each fit call takes regardless of dataset size. Unset (`None`) means use the whole dataset split.                                    |
| `MAX_EVAL_SEGMENTS`      | `4`     | Maximum number of segments used per evaluation call. Smaller values make evaluation fast; larger values give a more accurate loss estimate.                                                                      |
| `TRAIN_CHUNK_DURATION`   | `2.0`   | If set, each segment is processed in sub-chunks of this duration (seconds) during the forward/backward pass. Reduces peak memory when segments are long. Set equal to `SEGMENT_SECONDS` to disable sub-chunking. |
| `TRAIN_CHUNK_OVERLAP`    | `0.0`   | Fractional overlap between consecutive training sub-chunks, in `[0.0, 1.0)`. `0.0` = no overlap. Overlap can help at chunk boundaries but increases compute.                                                     |
| `EVAL_BATCH_SIZE`        | `1`     | Batch size used during evaluation. Higher values speed up eval but require more memory.                                                                                                                          |
| `USE_AMP`                | `false` | Enable PyTorch Automatic Mixed Precision (float16 forward pass, float32 gradients). Only effective on CUDA — ignored on CPU. Reduces GPU memory by ~40 % and speeds up training on supported hardware.           |
| `PREDICT_CHUNK_DURATION` | `30.0`  | Sub-chunk duration (seconds) used during inference/prediction on long files. Prevents OOM when running the model on full tracks.                                                                                 |

---

## Loss Weights

The composite loss is a weighted sum of five separation terms plus an optional classification term.

| Variable                  | Default | Description                                                                                                                                                                       |
|---------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LOSS_MASK_WEIGHT`        | `0.35`  | Weight for the soft mask loss (L1 between predicted and ideal ratio mask). Encourages the model to learn clean source masks in the spectrogram domain.                            |
| `LOSS_MAG_WEIGHT`         | `0.35`  | Weight for the magnitude loss (L1 between estimated and reference magnitude spectrograms). Penalises errors in the reconstructed source magnitudes directly.                      |
| `LOSS_CONSISTENCY_WEIGHT` | `0.05`  | Weight for the consistency loss, which penalises the model when the four stem magnitude estimates do not sum back to the mixture magnitude. Enforces energy conservation.         |
| `LOSS_KL_WEIGHT`          | `0.05`  | Weight for the KL-divergence regularisation term on the mask distribution. Prevents masks from being overconfident (all 0 or all 1).                                              |
| `LOSS_SISDR_WEIGHT`       | `0.20`  | Weight for the Scale-Invariant Signal-to-Distortion Ratio loss, computed in the waveform domain after iSTFT reconstruction. Directly optimises a perceptual audio quality metric. |
| `LOSS_CLS_WEIGHT`         | `0.10`  | Weight for the stem presence classification loss (BCE on the assigned stem's logit). Only applied when `TARGET_STEM` is set. Enables the acoustic perception capability. Set to `0.0` to disable classification training entirely. |

---

## Data Augmentation

Augmentation is applied only to the training split. All augmentation is disabled by default in the base app; the federated compose files (`docker-compose.federated.yaml`, `docker-compose.jetson-client.yaml`) override `AUGMENT_TRAIN=true` and `AUGMENT_STEM_DROPOUT_PROB=0.3`.

| Variable                    | Default | Description                                                                                                                                                                             |
|-----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `AUGMENT_TRAIN`             | `false` | Master switch. Must be `true` for any augmentation to apply.                                                                                                                            |
| `AUGMENT_GAIN_DB`           | `6.0`   | Maximum random gain perturbation in decibels. Each batch is scaled by a factor drawn uniformly from `[-gain_db, +gain_db]`. Improves robustness to loudness variation.                  |
| `AUGMENT_NOISE_STD`         | `0.0`   | Standard deviation of Gaussian noise added to the mixture waveform. `0.0` disables noise injection.                                                                                     |
| `AUGMENT_STEM_DROPOUT_PROB` | `0.0`   | Probability of zeroing out an individual stem before mixing. Teaches the model to handle missing sources. **Also controls the acoustic perception task difficulty**: when a stem is zeroed its spectrogram energy drops below the presence threshold, automatically generating a "not present" label. A value of `0.3` is recommended when training with `LOSS_CLS_WEIGHT > 0`. |
| `AUGMENT_SHIFT_SECONDS`     | `0.0`   | Maximum random time shift in seconds applied to each segment start. Adds variety without changing the underlying audio content.                                                         |
| `AUGMENT_REMIX_PROB`        | `0.0`   | Probability that any stem in a batch is swapped for the same stem from a randomly chosen different track. Creates synthetic mixtures not present in the original dataset.               |
| `AUGMENT_PHASE_FLIP`        | `false` | When `true`, randomly inverts the polarity (phase) of individual stems with 50 % probability. A cheap augmentation that is transparent to the human ear but increases spectral variety. |

---

## Acoustic Perception

The U-Net has a **dual head**: a separation mask head and a classification head (bottleneck → global average pool → linear) that outputs one logit per stem. During training, only the logit for `TARGET_STEM` is trained; all four logits are aggregated by FedAvg alongside the separation weights.

Presence labels are derived automatically from spectrogram energy: a stem is labelled *present* if its mean magnitude exceeds `1e-4`; stems zeroed by `AUGMENT_STEM_DROPOUT_PROB` fall below this threshold and are labelled *absent*.

**Metrics reported per round:**

| Metric         | Where      | Description                                                                              |
|----------------|------------|------------------------------------------------------------------------------------------|
| `cls_loss`     | fit / eval | BCE loss on the assigned stem's logit for this round.                                    |
| `cls_accuracy` | fit / eval | Fraction of samples where the sigmoid of the assigned stem's logit matched the label.    |

**HTTP endpoint:** `POST /external/classify` — accepts a WAV file, returns `{"stem": "vocals", "present": true, "confidence": 0.87}`. Scoped to `TARGET_STEM`; returns all four stems if unset.