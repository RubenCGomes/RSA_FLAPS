ML-App for FL Source Separation (prototype)

Overview
- MUSDB18HQ source separation ML-App built around a manifest-driven dataset loader.
- Uses a wider residual spectrogram U-Net that predicts four separation masks (vocals, drums, bass, other).
- Training combines softmax mask learning, multi-scale magnitude reconstruction, and mask-consistency regularization.
- Training and evaluation use batches, with SI-SDR computed from reconstructed waveforms.

Quick start (local prototype)

1. Create a Python environment and install dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements-mlapp.txt
    ```

2. Run the local smoke test:

    ```bash
    bash scripts/run_local.sh
    ```

3. Train a stronger model for longer:

    ```bash
    python scripts/train_musdb.py --epochs 10 --batch-size 1 --accumulation-steps 4 --max-eval-segments 32
    ```

    Or use an INI config file for most settings:
    
    ```bash
    python scripts/train_musdb.py --config scripts/config.ini
    ```

    CLI flags still override config values. Example (override only epochs):
    
    ```bash
    python scripts/train_musdb.py --config scripts/config.ini --epochs 2
    ```
    
    The shipped `scripts/config.ini` is intentionally minimal: it only overrides the
    few values that differ from the built-in defaults for the current full-track,
    chunked-training workflow.
    
    - By default, training uses the full `train` split. Add `--max-train-segments N` only if you want to cap the training subset for a quick experiment.

4. Record from the microphone and run prediction:

    ```bash
    python scripts/predict_from_mic.py --seconds 5 --sample-rate 16000 --checkpoint runs/train/<timestamp>/best.pt
    ```

5. Run prediction directly from an audio file:

```bash
python scripts/predict_from_file.py --audio-path path/to/mix.wav --checkpoint runs/train/<timestamp>/best.pt
```

What the smoke test does
- Builds a small train/val subset from `musdb18hq`
- Runs one short training epoch
- Evaluates on a tiny validation subset
- Runs prediction for one preview mixture and writes separated stems into `runs/`

What the microphone test does
- Records live audio from your default Linux input device
- Saves the captured input as `input.wav`
- Runs `MLApp.predict()` on the waveform
- Writes separated stems to a timestamped folder under `runs/mic/`
- If you train with a non-default model width, pass `--base-filters` to both the training and mic scripts so the checkpoint architecture matches.

What the file prediction script does
- Loads a local audio file from disk
- Saves predictions to a timestamped folder under `runs/file/`
- Calls `MLApp.predict()` with the file path directly

What the training CLI does
- Trains for multiple epochs on a larger MUSDB subset
- Evaluates every epoch on the validation split
- Saves `latest.pt` and `best.pt` in the run folder
- Stops early if validation SI-SDR improvement stops improving

Notes
- Default prototype settings use 16 kHz and 5-second segments.
- Dataset manifests are cached under `musdb18hq/.cache/`.
- You can override training size with `MAX_TRAIN_SEGMENTS`, `MAX_EVAL_SEGMENTS`, `CLIENT_ID`, and `NUM_CLIENTS` environment variables.
- For Linux microphone capture, you may need PortAudio system libraries first (for example `sudo apt-get install portaudio19-dev`).
