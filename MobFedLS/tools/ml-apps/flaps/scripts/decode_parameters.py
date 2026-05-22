#!/usr/bin/env python3
# decode_parameters.py
# Usage:
#   python decode_parameters.py --url "http://localhost:5001/internal/get_parameters?client_n=1"
#   python decode_parameters.py --file encoded_response.txt

import argparse
import json
import base64
import numpy as np
import sys
from urllib.parse import unquote
import requests

def maybe_unquote(s: str) -> str:
    # If string contains percent-encodings like %2F, unquote them.
    if '%' in s:
        return unquote(s)
    return s

def parse_json_maybe_string(payload: str):
    """
    The payload may be:
    - A JSON string already (--> parse directly).
    - A URL-encoded JSON string (--> unquote then parse).
    - Or the server may have wrapped the JSON into a JSON-string (double quotes around JSON).
    This function tries to robustly parse one or two levels.
    """
    txt = payload.strip()
    # first try direct JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # try unquoting then JSON
    try:
        txt2 = maybe_unquote(txt)
        return json.loads(txt2)
    except Exception:
        pass
    # sometimes the body is a quoted JSON string -> strip outer quotes and unescape
    if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
        inner = txt[1:-1]
        try:
            return json.loads(unquote(inner))
        except Exception:
            pass
    raise ValueError("Unable to parse payload as JSON (tried unquote & stripping quotes).")

def decode_one_param(param_obj):
    """
    param_obj: {"data":"<base64>", "shape":[...], "dtype":"float32"}
    returns numpy.ndarray
    """
    if not isinstance(param_obj, dict):
        raise ValueError("Expected dict per-parameter, got: %r" % type(param_obj))
    data_b64 = param_obj.get("data")
    shape = param_obj.get("shape")
    dtype = param_obj.get("dtype")
    if data_b64 is None or shape is None or dtype is None:
        raise ValueError("Parameter object missing data/shape/dtype keys: %r" % param_obj.keys())
    raw = base64.b64decode(data_b64)
    # Map dtype string to numpy dtype
    np_dtype = np.dtype(dtype)
    arr = np.frombuffer(raw, dtype=np_dtype)
    arr = arr.reshape(tuple(shape))
    return arr

def decode_parameters_list(obj):
    """
    obj may be:
      - a list of parameter dicts
      - a dict with key "parameters" -> list OR URL-encoded JSON string
      - a dict mapping names->param-dicts (then we return ordered pairs)

    The /internal/get_parameters endpoint returns {"parameters": "<url-encoded-json>"},
    where the value is the output of encoder_decoder.encode_parameters() — a URL-encoded
    JSON list of {data, shape, dtype} objects.  This function handles that case.
    """
    if isinstance(obj, list):
        return [decode_one_param(p) for p in obj]
    if isinstance(obj, dict):
        for key in ("parameters", "params"):
            if key not in obj:
                continue
            val = obj[key]
            if isinstance(val, list):
                return [decode_one_param(p) for p in val]
            if isinstance(val, str):
                # The MFLS interface stores the parameter list as a URL-encoded JSON
                # string inside the JSON response (encode_parameters returns quote(json_dumps(...))).
                # Decode it with the same two-step: unquote → json.loads.
                inner = parse_json_maybe_string(val)
                if isinstance(inner, list):
                    return [decode_one_param(p) for p in inner]
        # maybe mapping name -> dict
        items = []
        try:
            for k, v in obj.items():
                if isinstance(v, dict) and "data" in v:
                    items.append((k, decode_one_param(v)))
            if items:
                return items
        except Exception:
            pass
    raise ValueError("Unrecognized parameter container format.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL to GET (e.g. http://localhost:5001/internal/get_parameters?client_n=1)")
    parser.add_argument("--file", help="File that contains the raw response text")
    parser.add_argument("--out-npy", help="Optional directory to save each param as .npy (prefix 'param_0.npy')", default=None)
    args = parser.parse_args()

    if not (args.url or args.file):
        print("Pass --url or --file", file=sys.stderr)
        sys.exit(2)

    if args.url:
        r = requests.get(args.url, timeout=30)
        r.raise_for_status()
        payload = r.text
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            payload = f.read()

    parsed = parse_json_maybe_string(payload)
    decoded = decode_parameters_list(parsed)

    # Print summary (do not dump full arrays)
    if isinstance(decoded, list) and decoded and isinstance(decoded[0], np.ndarray):
        print("Decoded list of %d parameter arrays." % len(decoded))
        for i, arr in enumerate(decoded):
            print(f"param[{i}]: shape={arr.shape}, dtype={arr.dtype}, mean={float(np.mean(arr)):.6g}, std={float(np.std(arr)):.6g}")
            if args.out_npy:
                np.save(f"{args.out_npy}/param_{i}.npy", arr)
    else:
        # maybe items list (name, arr)
        print("Decoded mapping-style parameters:")
        for k,v in decoded:
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, mean={float(np.mean(v)):.6g}, std={float(np.std(v)):.6g}")
            if args.out_npy:
                np.save(f"{args.out_npy}/{k}.npy", v)

if __name__ == "__main__":
    main()