#!/usr/bin/env bash
# Simple helper to run the ML-App in local debug mode
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH=$ROOT_DIR python3 - <<'PY'
from ml_app.mfls_app import MLApp

app = MLApp(max_train_segments=4, max_eval_segments=2)
info = app.get_data({"max_segments": 4, "max_eval_segments": 2})
print('Data info:', info)

params = app.get_parameters()
print('Params count:', len(params))

trained_params, num_examples, train_metrics = app.fit(params, {"epochs": 1, "batch_size": 1, "accumulation_steps": 2, "max_segments": 4, "use_amp": True})
print('Train:', {'num_examples': num_examples, 'metrics': train_metrics})

eval_metrics = app.evaluate(trained_params, {"split": "val", "max_segments": 2, "batch_size": 1, "use_amp": True})
print('Eval:', eval_metrics)

preview = info['preview'][0]['files']['mixture']
preds = app.predict(trained_params, [preview])
print('Predict:', preds)
PY
