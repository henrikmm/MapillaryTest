"""Shared helpers: config loading, path resolution, logging, geo utilities."""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


# --------------------------------------------------------------------------- config

@dataclass
class Config:
    raw: dict
    config_path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

    def path(self, *keys: str) -> Path:
        """Resolve a path entry from `paths.*`, expanding relative paths against exp_root."""
        node = self.raw["paths"]
        for k in keys:
            node = node[k]
        p = Path(os.path.expanduser(str(node)))
        if not p.is_absolute():
            p = (self.exp_root / p).resolve()
        return p

    @property
    def exp_root(self) -> Path:
        root = Path(os.path.expanduser(str(self.raw["paths"]["exp_root"])))
        if not root.is_absolute():
            root = (self.config_path.parent / root).resolve()
        return root

    @property
    def outputs_dir(self) -> Path:
        return self.path("outputs_dir")


def load_config(path: str | Path) -> Config:
    p = Path(path).resolve()
    with p.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, config_path=p)


# --------------------------------------------------------------------------- logging

def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mfs")


# --------------------------------------------------------------------------- io

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def _json_default(o):
    # numpy scalars
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


# --------------------------------------------------------------------------- geo

_EARTH_R = 6_378_137.0  # WGS84 equatorial radius, m


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(a))


def lonlat_to_local_enu(
    lon: float, lat: float, alt: float, lon0: float, lat0: float, alt0: float
) -> tuple[float, float, float]:
    """
    Local tangent-plane (ENU) approximation around (lon0, lat0, alt0). Adequate
    for sequences spanning < a few km — at that scale the approximation error is
    well under 1 m. Used as the SfM topocentric reference frame.
    """
    lat0_rad = math.radians(lat0)
    east = math.radians(lon - lon0) * _EARTH_R * math.cos(lat0_rad)
    north = math.radians(lat - lat0) * _EARTH_R
    up = alt - alt0
    return east, north, up
