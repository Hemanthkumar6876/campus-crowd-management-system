"""Synthetic campus crowd generator.

Used to seed the historical database so the prediction and evaluation modules
work out of the box, and to simulate a live feed from cameras during demos.
Replace/extend it with real camera counts as they are collected.
"""
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd

import database as db
from config import LOCATIONS

# (peak hour, width in hours, amplitude as fraction of capacity)
PROFILES = {
    "Main Gate": [(8.5, 0.8, 0.55), (13.5, 0.7, 0.35), (17.0, 0.9, 0.55)],
    "Library": [(11.0, 2.0, 0.45), (15.0, 2.5, 0.50), (20.0, 1.5, 0.35)],
    "Cafeteria": [(10.5, 0.6, 0.30), (13.0, 0.7, 0.95), (16.5, 0.7, 0.35)],
    "Admin Block": [(11.0, 1.5, 0.50), (15.0, 1.5, 0.40)],
    "Lecture Hall Complex": [(h, 0.25, 0.65) for h in (8.9, 9.9, 10.9, 11.9, 13.9, 14.9, 15.9)],
    "Sports Ground": [(7.0, 1.0, 0.40), (17.0, 1.3, 0.70)],
    "Hostel Entrance": [(7.5, 0.7, 0.50), (13.5, 0.6, 0.35), (18.5, 1.2, 0.55), (21.5, 1.0, 0.50)],
    "Parking Area": [(8.5, 1.0, 0.60), (17.2, 0.9, 0.60)],
}
BASE = 0.03


@lru_cache(maxsize=None)
def _day_factor(loc, day):
    rng = np.random.default_rng(zlib.crc32(f"{loc}{day}".encode()))
    return rng.uniform(0.85, 1.15)


@lru_cache(maxsize=None)
def _event(day):
    """Occasional campus event creating a surge at one location."""
    rng = np.random.default_rng(zlib.crc32(f"event{day}".encode()))
    if rng.random() < 0.5:
        return list(LOCATIONS)[rng.integers(len(LOCATIONS))], rng.uniform(10, 18), 0.55
    return None


def expected_ratio(loc, ts):
    h = ts.hour + ts.minute / 60
    r = BASE + sum(a * np.exp(-((h - c) ** 2) / (2 * w * w)) for c, w, a in PROFILES[loc])
    weekend = ts.weekday() >= 5
    if weekend:
        r *= 0.4
    r *= _day_factor(loc, ts.date().isoformat())
    ev = _event(ts.date().isoformat())
    if ev and not weekend and ev[0] == loc:
        r += ev[2] * np.exp(-((h - ev[1]) ** 2) / (2 * 0.5**2))
    return r


def simulate_count(loc, ts, rng):
    ratio = expected_ratio(loc, ts) * rng.normal(1, 0.06)
    return int(rng.poisson(max(ratio, 0) * LOCATIONS[loc]["capacity"]))


def seed_history(days=14, end=None, step_min=5, seed=42):
    end = pd.Timestamp(end or pd.Timestamp.now()).floor(f"{step_min}min")
    idx = pd.date_range(end - pd.Timedelta(days=days), end, freq=f"{step_min}min")
    rng = np.random.default_rng(seed)
    rows = [(ts, loc, simulate_count(loc, ts, rng), "simulated") for ts in idx for loc in LOCATIONS]
    return db.add_observations(rows)


def simulate_next(last_ts, steps=1, step_min=5, rng=None):
    """Advance the simulated live feed; returns the rows written."""
    rng = rng or np.random.default_rng()
    rows = []
    for k in range(1, steps + 1):
        ts = pd.Timestamp(last_ts) + pd.Timedelta(minutes=step_min * k)
        for loc in LOCATIONS:
            rows.append((ts, loc, simulate_count(loc, ts, rng), "simulated"))
    db.add_observations(rows)
    return rows
