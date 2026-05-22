#!/usr/bin/env python3
"""distribute_song.py — central node orchestration script.

Sends a full audio mix to every FL client node, waits for each to separate
its assigned stem, then triggers synchronized playback across all nodes.

Usage
-----
  # Synchronized playback (default) — all nodes start playing at the same time:
  python distribute_song.py --song mix.wav \\
      --clients http://192.168.1.10:5000 http://192.168.1.11:5000

  # Skip synchronization — each node plays as soon as it finishes separating:
  python distribute_song.py --song mix.wav \\
      --clients http://192.168.1.10:5000 http://192.168.1.11:5000 --no-sync

  # Stop playback on all nodes:
  python distribute_song.py --stop \\
      --clients http://192.168.1.10:5000 http://192.168.1.11:5000
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import requests

TIMEOUT_INFO = 5
TIMEOUT_SEPARATE = 300   # separation can take a while on large files
TIMEOUT_PLAYBACK = 10


def _get(url: str, timeout: int = TIMEOUT_INFO) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _post(url: str, timeout: int = TIMEOUT_PLAYBACK, **kwargs) -> dict | None:
    r = requests.post(url, timeout=timeout, **kwargs)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None


def _delete(url: str, timeout: int = TIMEOUT_PLAYBACK) -> dict | None:
    r = requests.delete(url, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def discover(client_url: str) -> dict:
    return _get(f"{client_url}/external/separate/info")


def buffer_song(client_url: str, song_path: str) -> dict:
    with open(song_path, "rb") as f:
        return _post(
            f"{client_url}/external/separate/buffer",
            timeout=TIMEOUT_SEPARATE,
            files={"file": (song_path, f, "audio/wav")},
        )


def separate_and_stream(client_url: str, song_path: str, out_path: str) -> None:
    """Separate without buffering — streams the WAV back and saves it locally."""
    with open(song_path, "rb") as f:
        r = requests.post(
            f"{client_url}/external/separate",
            timeout=TIMEOUT_SEPARATE,
            files={"file": (song_path, f, "audio/wav")},
            stream=True,
        )
    r.raise_for_status()
    with open(out_path, "wb") as out:
        for chunk in r.iter_content(chunk_size=65536):
            out.write(chunk)


def trigger_playback(client_url: str) -> dict:
    return _post(f"{client_url}/external/playback")


def stop_playback(client_url: str) -> dict:
    return _delete(f"{client_url}/external/playback")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_parallel(fn, urls: list[str], label: str, **kwargs) -> dict[str, Future]:
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = {url: pool.submit(fn, url, **kwargs) for url in urls}
        results = {}
        for url, fut in futures.items():
            try:
                results[url] = fut.result()
                print(f"  ✓ {url}")
            except Exception as exc:
                print(f"  ✗ {url} — {exc}")
                results[url] = None
        return results


def _all_ok(results: dict) -> bool:
    return all(v is not None for v in results.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribute a song to FL client nodes for stem separation and playback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--song", help="Path to the full audio mix (WAV/MP3/FLAC)")
    parser.add_argument(
        "--clients",
        nargs="+",
        required=True,
        metavar="URL",
        help="Client base URLs, e.g. http://192.168.1.10:5000",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Each node plays immediately after separating instead of waiting for all nodes",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop playback on all nodes and exit",
    )
    args = parser.parse_args()

    if args.stop:
        print("Stopping playback on all nodes...")
        _run_parallel(stop_playback, args.clients, "stop")
        return

    if not args.song:
        parser.error("--song is required unless --stop is used")

    # ------------------------------------------------------------------
    # Step 1: discover
    # ------------------------------------------------------------------
    print("Discovering clients...")
    infos: dict[str, dict] = {}
    for url in args.clients:
        try:
            info = discover(url)
            infos[url] = info
            stem = info.get("target_stem") or "all stems"
            busy = info.get("busy", False)
            print(f"  ✓ {url} — stem: {stem}{' (BUSY — FL in progress)' if busy else ''}")
            if busy:
                print("    Aborting: cannot separate while FL is running.")
                sys.exit(1)
        except Exception as exc:
            print(f"  ✗ {url} — unreachable: {exc}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: send song and separate
    # ------------------------------------------------------------------
    sync = not args.no_sync

    if sync:
        print(f"\nSending '{args.song}' to {len(args.clients)} client(s) (buffered)...")
        with ThreadPoolExecutor(max_workers=len(args.clients)) as pool:
            futures = {url: pool.submit(buffer_song, url, args.song) for url in args.clients}
            ok = True
            for url, fut in futures.items():
                try:
                    result = fut.result()
                    stem = result.get("stem", "?")
                    print(f"  ✓ {url} — '{stem}' buffered")
                except Exception as exc:
                    print(f"  ✗ {url} — {exc}")
                    ok = False
            if not ok:
                sys.exit(1)

        # ------------------------------------------------------------------
        # Step 3: fire playback simultaneously
        # ------------------------------------------------------------------
        print("\nStarting synchronized playback on all nodes...")
        with ThreadPoolExecutor(max_workers=len(args.clients)) as pool:
            futures = {url: pool.submit(trigger_playback, url) for url in args.clients}
            for url, fut in futures.items():
                try:
                    result = fut.result()
                    stem = result.get("stem", "?")
                    print(f"  ✓ {url} — playing '{stem}'")
                except Exception as exc:
                    print(f"  ✗ {url} — {exc}")

    else:
        # Non-sync: stream the WAV back to the orchestrator and save locally
        print(f"\nSending '{args.song}' to {len(args.clients)} client(s) (streaming, no sync)...")
        with ThreadPoolExecutor(max_workers=len(args.clients)) as pool:
            futures = {}
            for url in args.clients:
                stem = infos[url].get("target_stem") or "stems"
                out_path = f"{stem}_from_{url.split('/')[-1].replace(':', '_')}.wav"
                futures[url] = (pool.submit(separate_and_stream, url, args.song, out_path), out_path)
            for url, (fut, out_path) in futures.items():
                try:
                    fut.result()
                    print(f"  ✓ {url} — saved to {out_path}")
                except Exception as exc:
                    print(f"  ✗ {url} — {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()