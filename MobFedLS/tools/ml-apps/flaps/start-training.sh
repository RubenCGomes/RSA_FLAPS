#!/bin/bash
# Start federated learning training with FLAPS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
N_ROUNDS=${N_ROUNDS:-5}
N_CLIENTS=${N_CLIENTS:-1}
N_EPOCHS=${N_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
FL_ALGORITHM=${FL_ALGORITHM:-FedAvg}
TRAIN_CHUNK_DURATION=${TRAIN_CHUNK_DURATION:-3.0}
USE_AMP=${USE_AMP:-true}
LOGGING_LEVEL=${LOGGING_LEVEL:-INFO}

echo "========================================="
echo "FLAPS Federated Learning Training"
echo "========================================="
echo "Configuration:"
echo "  Rounds:           $N_ROUNDS"
echo "  Clients:          $N_CLIENTS"
echo "  Epochs/round:     $N_EPOCHS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Algorithm:        $FL_ALGORITHM"
echo "  Chunk duration:   ${TRAIN_CHUNK_DURATION}s"
echo "  AMP:              $USE_AMP"
echo "========================================="
echo ""

# Start federated learning stack
N_ROUNDS="$N_ROUNDS" \
N_CLIENTS="$N_CLIENTS" \
N_EPOCHS="$N_EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
FL_ALGORITHM="$FL_ALGORITHM" \
TRAIN_CHUNK_DURATION="$TRAIN_CHUNK_DURATION" \
USE_AMP="$USE_AMP" \
LOGGING_LEVEL="$LOGGING_LEVEL" \
docker compose -f docker-compose.federated.yaml up --build

# Note: Use Ctrl+C to stop the training

