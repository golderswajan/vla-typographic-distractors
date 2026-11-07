"""Analysis helpers for the typographic-injection proof of concept (no torch needed)."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from PIL import Image

from vla_attack_patch import box_mask_on_grid


def load_condition(log_dir: str, eval_dir: str) -> list[dict]:
    """Load per-episode attention logs and attach success flags from lerobot's eval_info.json."""
    files = sorted(glob.glob(os.path.join(log_dir, "ep*.npz")))
    episodes = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        meta["maps"] = z["maps"]
        meta["frames"] = z["frames"]
        episodes.append(meta)
    episodes.sort(key=lambda e: e["episode"])

    info_path = os.path.join(eval_dir, "eval_info.json")
    with open(info_path) as fh:
        info = json.load(fh)
    successes = []
    for t in info["per_task"]:
        successes.extend(bool(s) for s in t["metrics"]["successes"])
    if len(successes) != len(episodes):
        print(f"WARNING {log_dir}: {len(episodes)} logged episodes vs {len(successes)} results; aligning by order")
    for e, s in zip(episodes, successes):
        e["success"] = s
    return [e for e in episodes if "success" in e]


def box_attention(ep: dict) -> dict:
    """Share of main-camera attention that falls on the label region (averaged over chunks)."""
    maps = ep["maps"]
    if maps.ndim != 3 or maps.shape[0] == 0 or maps.shape[1] < 2:
        return {"box_mass": np.nan, "box_ratio": np.nan, "n_chunks": 0}
    side = maps.shape[1]
    mask = box_mask_on_grid(side, ep["box"])
    norm = maps / np.clip(maps.sum(axis=(1, 2), keepdims=True), 1e-12, None)
    mass = norm[:, mask].sum(axis=1)  # per chunk
    area = mask.mean() if mask.any() else np.nan
    return {
        "box_mass": float(mass.mean()),
        "box_mass_first3": float(mass[:3].mean()),  # early-warning version
        "box_ratio": float(mass.mean() / area) if area == area else np.nan,
        "n_chunks": int(maps.shape[0]),
    }


def heat_overlay(frame: np.ndarray, amap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a (side x side) attention map onto an HxWx3 frame (jet-like colours, no matplotlib)."""
    h, w = frame.shape[:2]
    a = amap / max(amap.max(), 1e-12)
    up = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    r = np.clip(1.5 - np.abs(4 * up - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * up - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * up - 1), 0, 1)
    heat = (np.stack([r, g, b], -1) * 255).astype(np.float32)
    return ((1 - alpha) * frame.astype(np.float32) + alpha * heat).clip(0, 255).astype(np.uint8)


def map_for_frame(ep: dict, frame_step: int) -> np.ndarray | None:
    """Latest attention map computed from an observation at or before `frame_step`."""
    best = None
    for m, s in zip(ep["maps"], ep["map_steps"]):
        if s - 1 <= frame_step:  # map_steps are logged one step after the observation they used
            best = m
    return best


def episode_gif(ep: dict, path: str, with_heat: bool = True, fps: int = 6) -> str | None:
    frames = ep["frames"]
    if frames.ndim != 4 or len(frames) == 0:
        return None
    out = []
    for fr, st in zip(frames, ep["frame_steps"]):
        m = map_for_frame(ep, st) if with_heat else None
        img = heat_overlay(fr, m) if m is not None else fr
        out.append(Image.fromarray(img).resize((256, 256)))
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(1000 / fps), loop=0)
    return path


def auroc(labels, scores) -> float:
    """Mann-Whitney AUROC without sklearn (ties count half)."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    ok = ~np.isnan(scores)
    labels, scores = labels[ok], scores[ok]
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
