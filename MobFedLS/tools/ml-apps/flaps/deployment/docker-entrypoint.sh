#!/bin/sh
# Entrypoint: ensure dataset/logs directories and writable cache exist, then start app
set -e

echo "[entrypoint] Checking dataset and logs directories..."
mkdir -p /app/dataset
mkdir -p /app/logs

# try to create .cache to avoid runtime write errors when host mount exists
if [ -d /app/dataset ]; then
  if [ ! -d /app/dataset/.cache ]; then
    mkdir -p /app/dataset/.cache || true
  fi
  # attempt to set permissive permissions to allow writes from container
  chmod -R a+rwx /app/dataset || true
fi

if [ -d /app/logs ]; then
  chmod -R a+rwx /app/logs || true
fi

echo "[entrypoint] Starting ML app (main.py if present, fallback to internal interface)..."

# Prefer main.py if present in /app
if [ -f /app/main.py ]; then
  exec python3 /app/main.py
elif [ -f /app/clientML.py ] && [ -d /app/ml_app ]; then
  # If the project contains the mfls interface module, try to launch the internal interface
  if [ -f /app/internal/mfls-interface/cmd/main.py ]; then
    exec python3 /app/internal/mfls-interface/cmd/main.py
  fi
  # Fallback: try running any packaging-provided main
  if python3 -c "import importlib, sys; importlib.import_module('internal.mfls_interface.cmd.main')" 2>/dev/null; then
    exec python3 -m internal.mfls_interface.cmd.main
  fi
  # Last resort: try to run uvicorn pointing to the interface if installed
  if python3 -c "import uvicorn" 2>/dev/null; then
    exec uvicorn "interface.interface:mflinterface_app" --host 0.0.0.0 --port 5000
  fi
fi

echo "[entrypoint] Could not find a known start command; launching 'python3 -u -m http.server 5000' as debug fallback"
exec python3 -u -m http.server 5000

