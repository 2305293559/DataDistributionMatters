# random_sampling_10d.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

DomainType = Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]], Tuple[Tuple[float, float], ...]]

def _normalize_domain(domain: DomainType) -> Tuple[List[str], List[Tuple[float, float]]]:
    if isinstance(domain, dict):
        names = list(domain.keys())
        bounds = [tuple(domain[k]) for k in names]
    elif isinstance(domain, (list, tuple)):
        bounds = [tuple(b) for b in domain]
        names = [f"x{i+1}" for i in range(len(bounds))]
    else:
        raise ValueError("domain must be dict or list/tuple of (low, high) pairs.")
    for b in bounds:
        if len(b) != 2:
            raise ValueError("each bound must be a (low, high) pair.")
        if b[0] >= b[1]:
            raise ValueError(f"invalid bound: low >= high {b}")
    return names, bounds

def generate_random_samples(
    domain: DomainType,
    n_samples: int = 50,
    decimals: Optional[int] = 3,
    seed: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    names, bounds = _normalize_domain(domain)
    dim = len(bounds)

    if columns is not None:
        if len(columns) != dim:
            raise ValueError("length of columns must equal domain dimension.")
        col_names = columns
    else:
        col_names = names

    samples = np.empty((n_samples, dim), dtype=float)
    for j, (low, high) in enumerate(bounds):
        samples[:, j] = np.random.uniform(low=low, high=high, size=n_samples)

    if decimals is not None:
        samples = np.round(samples, decimals)

    return pd.DataFrame(samples, columns=col_names)

def save_samples_to_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    df.to_csv(filename, index=index)

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Generate 10D random samples and save to CSV.")
    parser.add_argument("-n", "--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--decimals", type=int, default=None)
    parser.add_argument("-o", "--out", type=str, default="random_samples_10d.csv")
    parser.add_argument("--domain_json", type=str, default=None)
    args = parser.parse_args()

    default_domain = {
        "x1": (-1, 1),
        "x2": (-1, 1),
        "x3": (-1, 1),
        "x4": (-1, 1),
        "x5": (-1, 1),
        "x6": (-1, 1),
        "x7": (-1, 1),
        "x8": (-1, 1),
        "x9": (-1, 1),
        "x10":(-1, 1)
    }
    
    if args.domain_json:
        try:
            parsed = json.loads(args.domain_json)
            if isinstance(parsed, dict):
                domain = {k: tuple(v) for k, v in parsed.items()}
            elif isinstance(parsed, list):
                domain = [tuple(v) for v in parsed]
            else:
                raise ValueError("domain_json must be dict or list.")
        except Exception as e:
            print(f"failed to parse domain_json: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        domain = default_domain

    decimals = None if args.decimals == -1 else args.decimals

    df = generate_random_samples(domain, n_samples=args.n_samples, decimals=decimals, seed=args.seed)
    save_samples_to_csv(df, args.out)
    print(f"Generated {len(df)} samples and saved to {args.out}")
