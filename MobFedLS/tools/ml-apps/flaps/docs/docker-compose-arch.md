## Architecture:

```
  MUSDB18HQ dataset (shared mount)
         │
         ├── ml-app-1 (CLIENT_ID=0) ← ~half the tracks (hash % 2 == 0)
         │       ↑ HTTP
         │   flower-client-1 ──┐
         │                     │ gRPC
         └── ml-app-2 (CLIENT_ID=1) ← other half (hash % 2 == 1)
                 ↑ HTTP        │
             flower-client-2 ──┤
                               ▼
                         flower-server  ← FedAvg aggregation
```

Each client trains on its own disjoint track subset. After each round the server averages both clients' weights — that's the actual federation step. Aggregated weights are pushed back down to both clients for the next round.

## Commands:

```sh
# Start ml-apps idle (ports 5001 and 5002):
docker compose -f docker-compose.federated.yaml up

# Start FL training on top:
docker compose -f docker-compose.federated.yaml --profile training up

# Override number of clients (must match N_CLIENTS on server and ghost clients):
N_CLIENTS=3 docker compose -f docker-compose.federated.yaml up
# (you'd add ml-app-3 / flower-client-3 manually for 3)
```

Logs for each client land in `assets/logs/client-1/` and `assets/logs/client-2/` separately so you can compare their training curves.
