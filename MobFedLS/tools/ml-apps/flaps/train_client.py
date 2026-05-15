#!/usr/bin/env python3
"""
Python client for interacting with FLAPS ML app

Usage:
    python train_client.py --url http://localhost:5001 --action fit --epochs 2
    python train_client.py --url http://localhost:5001 --action evaluate
    python train_client.py --url http://localhost:5001 --action get-params
"""

import json
import argparse
import requests
import sys
from typing import Dict, Any, Optional


class FLAPSClient:
    """Client for FLAPS ML app interface"""

    def __init__(self, url: str = "http://localhost:5001"):
        self.url = url
        self.session = requests.Session()

    def get_parameters(self) -> Optional[Dict[str, Any]]:
        """Get current model parameters"""
        try:
            resp = self.session.get(f"{self.url}/internal/get_parameters")
            resp.raise_for_status()
            print("✓ Successfully retrieved parameters")
            print(f"  Response size: {len(resp.text)} bytes")
            return resp.json()
        except Exception as e:
            print(f"✗ Failed to get parameters: {e}")
            return None

    def get_data(self, client_n: int = 1) -> Optional[Dict[str, Any]]:
        """Get data info for a client"""
        try:
            resp = self.session.get(f"{self.url}/internal/get_data", params={"client_n": client_n})
            resp.raise_for_status()
            data_info = resp.json()
            print(f"✓ Got data info for client {client_n}")
            print(f"  Training samples: {data_info.get('num_train_samples', 'N/A')}")
            print(f"  Test samples: {data_info.get('num_test_samples', 'N/A')}")
            return data_info
        except Exception as e:
            print(f"✗ Failed to get data: {e}")
            return None

    def fit(self, epochs: int = 1, batch_size: int = 32, learning_rate: float = 0.001) -> bool:
        """Start a training round"""
        config = {
            "num_rounds": 1,
            "local_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        try:
            resp = self.session.post(
                f"{self.url}/internal/fit",
                json=config,
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            result = resp.json()
            print(f"✓ Training completed")
            print(f"  Samples: {result.get('num_examples', 'N/A')}")
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            return True
        except Exception as e:
            print(f"✗ Training failed: {e}")
            return False

    def evaluate(self, batch_size: int = 32) -> bool:
        """Evaluate the model"""
        try:
            resp = self.session.get(
                f"{self.url}/internal/evaluate",
                params={"batch_size": batch_size}
            )
            resp.raise_for_status()
            result = resp.json()
            print(f"✓ Evaluation completed")
            print(f"  Samples: {result.get('num_examples', 'N/A')}")
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            return True
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            return False

    def predict(self) -> bool:
        """Run prediction on a test sample"""
        try:
            resp = self.session.get(f"{self.url}/internal/predict")
            resp.raise_for_status()
            result = resp.json()
            print(f"✓ Prediction completed")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, list):
                        print(f"  {key}: list of {len(value)} items")
                    else:
                        print(f"  {key}: {value}")
            return True
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            return False

    def health_check(self) -> bool:
        """Check if the app is healthy"""
        try:
            resp = self.session.get(f"{self.url}/docs")
            if resp.status_code == 200:
                print(f"✓ App is healthy and running")
                return True
            else:
                print(f"✗ App returned status {resp.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="FLAPS ML App Training Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_client.py --url http://localhost:5001 check
  python train_client.py --url http://localhost:5001 get-params
  python train_client.py --url http://localhost:5001 get-data
  python train_client.py --url http://localhost:5001 fit --epochs 2 --batch-size 16
  python train_client.py --url http://localhost:5001 evaluate
  python train_client.py --url http://localhost:5001 predict
        """
    )
    parser.add_argument("action", help="Action to perform: check, get-params, get-data, fit, evaluate, predict")
    parser.add_argument("--url", default="http://localhost:5001", help="App URL (default: http://localhost:5001)")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for training (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--client-n", type=int, default=1, help="Client ID for get-data (default: 1)")

    args = parser.parse_args()

    client = FLAPSClient(args.url)

    print(f"\n{'='*50}")
    print(f"FLAPS Training Client")
    print(f"{'='*50}")
    print(f"App URL: {args.url}\n")

    success = False
    if args.action == "check":
        success = client.health_check()
    elif args.action == "get-params":
        success = client.get_parameters() is not None
    elif args.action == "get-data":
        success = client.get_data(args.client_n) is not None
    elif args.action == "fit":
        print(f"Configuration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}\n")
        success = client.fit(args.epochs, args.batch_size, args.learning_rate)
    elif args.action == "evaluate":
        print(f"Configuration:")
        print(f"  Batch size: {args.batch_size}\n")
        success = client.evaluate(args.batch_size)
    elif args.action == "predict":
        success = client.predict()
    else:
        print(f"Unknown action: {args.action}")
        parser.print_help()
        sys.exit(1)

    print()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

