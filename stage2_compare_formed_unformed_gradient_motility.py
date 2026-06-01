#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Gradient Motility of Stage-1 Formed vs Unformed Models
===============================================================

Corrected central comparison
----------------------------
This script first runs the Stage-1 membrane-thickness self-selection model.
Then it classifies each resulting model as:

    unformed:
        no lipid membrane patch at B >= 0.20

    formed_nonintermediate:
        lipid membrane patch exists, but no patch in the intermediate-thickness band

    formed_intermediate:
        lipid membrane patch exists and at least one preformed patch has:
            0.05 <= mean_T < 0.15

Only after this Stage-1 classification does the script place the resulting
model state into a common chemical-gradient world and observe motion.

Therefore the comparison is:

    Stage-1 membrane-formed models
    vs
    Stage-1 membrane-unformed models

in the same type of concentration-gradient environment.

No model is instructed to move, explore, survive, approach resources, maintain
a target thickness, or increase chi.

Hard exclusions
---------------
No goal.
No reward.
No reinforcement learning.
No optimizer.
No weight update.
No survival predictor.
No set-point.
No homeostatic objective.
No instruction to explore.
No instruction to move toward resource.
No target gradient alignment.
No target membrane thickness.
No target chi.
No success condition.
No parameter search for success.

Phases
------
1. Stage-1 preformation:
   Same core physical model as stage1_membrane_thickness_self_selection.py:
   microtopography, temporal retention, lipid precursor, membrane density B,
   membrane thickness T, permeability-thickness tradeoff.

2. Formation classification:
   Based only on observed Stage-1 state at the end of preformation:
   B-thresholded patch existence and patch mean thickness.

3. Common chemical-gradient exposure:
   Introduce R/L/H/X concentration gradients.
   Continue the same physical equations with chemical-potential-driven
   membrane material transport.

4. Observation:
   Track preformed patches by overlap and also track bulk membrane-material
   center of mass for every model, including unformed/diffuse cases.

Variants
--------
flat_temporal_lipid
microtopography_temporal_lipid
microtopography_lipid_no_temporal
microtopography_temporal_no_lipid
microtopography_shuffled_temporal_lipid

Outputs
-------
preformation_status.csv
preformation_patch_metrics.csv
preformed_patch_motility_tracks.csv
bulk_material_motility.csv
gradient_timeseries_metrics.csv
motility_summary_by_variant.csv
motility_summary_by_stage1_status.csv
motility_summary_by_onset_thickness_bin.csv
stage2_formed_unformed_integrated_report.txt
manifest.json
stage2_compare_formed_unformed_gradient_motility_outputs.zip
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


VARIANTS = [
    "flat_temporal_lipid",
    "microtopography_temporal_lipid",
    "microtopography_lipid_no_temporal",
    "microtopography_temporal_no_lipid",
    "microtopography_shuffled_temporal_lipid",
]

TRACK_THRESHOLD = 0.20
INTERMEDIATE_LO = 0.05
INTERMEDIATE_HI = 0.15
THICKNESS_BINS = [-1e-9, 0.05, 0.15, 0.30, 0.50, 0.75, 1.00, 2.50]
THICKNESS_LABELS = ["0-0.05", "0.05-0.15", "0.15-0.30", "0.30-0.50", "0.50-0.75", "0.75-1.00", "1.00+"]


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_mean(xs) -> float:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else 0.0


def safe_sd(xs) -> float:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def safe_q(xs, q: float) -> float:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.quantile(vals, q)) if vals else 0.0


def summarize(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = len(g)
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = safe_mean(g[m])
                row[f"{m}_sd"] = safe_sd(g[m])
                row[f"{m}_q25"] = safe_q(g[m], 0.25)
                row[f"{m}_q50"] = safe_q(g[m], 0.50)
                row[f"{m}_q75"] = safe_q(g[m], 0.75)
                row[f"{m}_max"] = float(np.max(g[m])) if len(g) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def sigmoid_array(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def grad(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = 0.5 * (np.roll(A, -1, axis=0) - np.roll(A, 1, axis=0))
    gy = 0.5 * (np.roll(A, -1, axis=1) - np.roll(A, 1, axis=1))
    return gx, gy


def grad_mag(A: np.ndarray) -> np.ndarray:
    gx, gy = grad(A)
    return np.sqrt(gx * gx + gy * gy)


def neighbor_mean(A: np.ndarray) -> np.ndarray:
    return 0.25 * (
        np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
        + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
    )


def variable_diffusion(A: np.ndarray, D: np.ndarray) -> np.ndarray:
    De = 0.5 * (D + np.roll(D, -1, axis=0))
    Dw = 0.5 * (D + np.roll(D, 1, axis=0))
    Dn = 0.5 * (D + np.roll(D, -1, axis=1))
    Ds = 0.5 * (D + np.roll(D, 1, axis=1))
    return (
        De * (np.roll(A, -1, axis=0) - A)
        + Dw * (np.roll(A, 1, axis=0) - A)
        + Dn * (np.roll(A, -1, axis=1) - A)
        + Ds * (np.roll(A, 1, axis=1) - A)
    )


def upwind_advection(A: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dx_up = np.where(u >= 0.0, A - np.roll(A, 1, axis=0), np.roll(A, -1, axis=0) - A)
    dy_up = np.where(v >= 0.0, A - np.roll(A, 1, axis=1), np.roll(A, -1, axis=1) - A)
    return -u * dx_up - v * dy_up


def autocorr_chi(series, dt: float, gamma: float) -> Tuple[float, float]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30 or np.std(x) < 1e-12:
        return 0.0, 0.0
    x = x - np.mean(x)
    var = np.sum(x * x)
    maxlag = max(5, min(len(x) // 3, 1200))
    acf = [float(np.sum(x[:len(x) - k] * x[k:]) / var) for k in range(maxlag + 1)]
    thr = 1.0 / math.e
    for k in range(1, len(acf)):
        if acf[k] <= thr:
            y0, y1 = acf[k - 1], acf[k]
            if abs(y1 - y0) < 1e-12:
                tc = k * dt
            else:
                tc = (k - 1 + (thr - y0) / (y1 - y0)) * dt
            return float(tc * gamma), float(tc)
    tc = maxlag * dt
    return float(tc * gamma), float(tc)


def connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or seen[i, j]:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    xx = (x + dx) % H
                    yy = (y + dy) % W
                    if mask[xx, yy] and not seen[xx, yy]:
                        seen[xx, yy] = True
                        q.append((xx, yy))
            comps.append(comp)
    return comps


def component_mask(shape, comp):
    m = np.zeros(shape, dtype=bool)
    for x, y in comp:
        m[x, y] = True
    return m


def center_of_mass(mask: np.ndarray, weight: np.ndarray) -> Tuple[float, float]:
    w = np.where(mask, np.maximum(weight, 1e-12), 0.0)
    total = float(np.sum(w))
    if total <= 0:
        xs, ys = np.where(mask)
        return (float(np.mean(xs)), float(np.mean(ys))) if len(xs) else (0.0, 0.0)
    xs = np.arange(mask.shape[0])[:, None]
    ys = np.arange(mask.shape[1])[None, :]
    return float(np.sum(xs * w) / total), float(np.sum(ys * w) / total)


def bulk_center(weight: np.ndarray) -> Tuple[float, float]:
    if float(np.sum(weight)) <= 1e-12:
        return 0.0, 0.0
    xs = np.arange(weight.shape[0])[:, None]
    ys = np.arange(weight.shape[1])[None, :]
    total = float(np.sum(weight))
    return float(np.sum(xs * weight) / total), float(np.sum(ys * weight) / total)


class ProgressLogger:
    def __init__(self, outdir: Path):
        mkdir(outdir)
        self.t0 = time.time()
        self.path = outdir / "run_progress.log"
        self.path.write_text(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclasses.dataclass
class Config:
    mode: str
    outdir: Path
    seed: int
    replicates: int
    size: int
    preform_steps: int
    gradient_steps: int
    sample_stride: int
    resume: bool
    make_figures: bool
    save_fields: bool
    dt: float
    gamma: float
    beta: float
    tau: float
    sigma: float


@dataclasses.dataclass
class PatchTrack:
    track_id: int
    birth_step: int
    initial_cells: Set[Tuple[int, int]]
    current_cells: Set[Tuple[int, int]]
    onset_thickness: float
    onset_area_fraction: float
    onset_intermediate_band: int
    steps: List[int]
    cx: List[float]
    cy: List[float]
    area_fraction: List[float]
    mean_T: List[float]
    mean_B: List[float]
    permeability: List[float]
    M_inside: List[float]
    M_outside: List[float]
    R_inside: List[float]
    X_inside: List[float]
    gradient_alignment: List[float]
    step_displacement: List[float]
    local_gradient_norm: List[float]
    internal_circulation: List[float]

    def update(self, step: int, cells: Set[Tuple[int, int]], metrics: Dict[str, float]):
        if self.cx:
            dx = metrics["center_x"] - self.cx[-1]
            dy = metrics["center_y"] - self.cy[-1]
            disp = math.sqrt(dx * dx + dy * dy)
            gn = metrics["local_gradient_norm"]
            if disp > 1e-12 and gn > 1e-12:
                align = (dx * metrics["local_gradient_x"] + dy * metrics["local_gradient_y"]) / (disp * gn)
            else:
                align = 0.0
        else:
            disp = 0.0
            align = 0.0
        self.current_cells = cells
        self.steps.append(step)
        self.cx.append(metrics["center_x"])
        self.cy.append(metrics["center_y"])
        self.area_fraction.append(metrics["area_fraction"])
        self.mean_T.append(metrics["mean_T"])
        self.mean_B.append(metrics["mean_B"])
        self.permeability.append(metrics["mean_permeability"])
        self.M_inside.append(metrics["M_inside"])
        self.M_outside.append(metrics["M_outside"])
        self.R_inside.append(metrics["R_inside"])
        self.X_inside.append(metrics["X_inside"])
        self.local_gradient_norm.append(metrics["local_gradient_norm"])
        self.gradient_alignment.append(align)
        self.step_displacement.append(disp)
        self.internal_circulation.append(metrics["internal_circulation"])


def build_microtopography(N: int, rng: np.random.Generator, flat: bool) -> Dict[str, np.ndarray]:
    Xg, Yg = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if flat:
        return {
            "pit_depth": np.zeros((N, N)),
            "mineral": np.ones((N, N)) * 0.22,
            "residence": np.ones((N, N)) * 0.18,
            "flow_speed": np.ones((N, N)) * 0.10,
            "u": np.zeros((N, N)),
            "v": np.zeros((N, N)),
            "vent": np.ones((N, N)) * 0.08,
        }

    pit = np.zeros((N, N))
    mineral_bumps = np.zeros((N, N))
    for _ in range(max(6, int(N / 7))):
        cx, cy = rng.uniform(0, N), rng.uniform(0, N)
        sx, sy = rng.uniform(0.035 * N, 0.095 * N), rng.uniform(0.035 * N, 0.095 * N)
        amp = rng.uniform(0.60, 1.20)
        pit += amp * np.exp(-((Xg - cx) ** 2 / (2 * sx * sx) + (Yg - cy) ** 2 / (2 * sy * sy)))
        mineral_bumps += 0.45 * amp * np.exp(-((Xg - cx) ** 2 / (2 * (sx * 1.45) ** 2) + (Yg - cy) ** 2 / (2 * (sy * 1.45) ** 2)))

    pit = pit / max(1e-9, float(np.max(pit)))
    bed = -pit + 0.12 * mineral_bumps + rng.normal(0.0, 0.02, (N, N))
    slope = grad_mag(bed)
    slope = slope / max(1e-9, float(np.max(slope)))
    mineral = np.clip(0.20 + 0.50 * mineral_bumps + 0.30 * slope + 0.22 * pit, 0, 1)
    porosity = np.clip(0.10 + 0.75 * pit + 0.10 * rng.random((N, N)), 0, 1)
    slowdown = np.clip(1.0 - 0.78 * pit - 0.22 * porosity, 0.08, 1.0)
    u = (0.018 + 0.006 * rng.normal(size=(N, N))) * slowdown
    v = 0.006 * rng.normal(size=(N, N)) * slowdown
    flow = np.sqrt(u * u + v * v)
    flow_norm = flow / max(1e-9, float(np.max(flow)))
    residence = np.clip(0.10 + 0.70 * pit + 0.28 * porosity + 0.18 * mineral - 0.22 * flow_norm, 0, 1)
    vent = np.clip(0.04 + 0.45 * pit * mineral + 0.10 * rng.random((N, N)), 0, 1)
    return {
        "pit_depth": pit,
        "mineral": mineral,
        "residence": residence,
        "flow_speed": flow,
        "u": u,
        "v": v,
        "vent": vent,
    }


def source_fields(N: int, topo: Dict[str, np.ndarray], gradient: bool) -> Dict[str, np.ndarray]:
    Xg, Yg = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if gradient:
        Rsrc = 0.12 + 0.45 * np.exp(-((Xg - 0.78 * N) ** 2 / (2 * (0.18 * N) ** 2) + (Yg - 0.32 * N) ** 2 / (2 * (0.22 * N) ** 2))) + 0.18 * topo["vent"]
        Lsrc = 0.10 + 0.40 * np.exp(-((Xg - 0.25 * N) ** 2 / (2 * (0.22 * N) ** 2) + (Yg - 0.76 * N) ** 2 / (2 * (0.18 * N) ** 2))) + 0.18 * topo["pit_depth"] + 0.12 * topo["mineral"]
        Hsrc = 0.08 + 0.55 * np.exp(-((Xg - 0.62 * N) ** 2 / (2 * (0.16 * N) ** 2) + (Yg - 0.62 * N) ** 2 / (2 * (0.16 * N) ** 2))) + 0.22 * topo["vent"]
        Xsrc = 0.04 + 0.22 * np.exp(-((Xg - 0.18 * N) ** 2 / (2 * (0.20 * N) ** 2) + (Yg - 0.22 * N) ** 2 / (2 * (0.20 * N) ** 2)))
    else:
        Rsrc = np.ones((N, N)) * 0.22 + 0.08 * topo["vent"]
        Lsrc = np.ones((N, N)) * 0.20 + 0.05 * topo["pit_depth"] + 0.04 * topo["mineral"]
        Hsrc = np.ones((N, N)) * 0.20 + 0.04 * topo["vent"]
        Xsrc = np.ones((N, N)) * 0.06
    return {
        "R": np.clip(Rsrc, 0, 1),
        "L": np.clip(Lsrc, 0, 1),
        "H": np.clip(Hsrc, 0, 1),
        "X": np.clip(Xsrc, 0, 1),
    }


def compute_patch_metrics(mask, B, T, P, R, X, L, H, M, phi, phix, phiy):
    outside = ~mask
    Mabs = np.abs(M)
    cx, cy = center_of_mass(mask, B + T)
    gx = float(np.mean(phix[mask]))
    gy = float(np.mean(phiy[mask]))
    return {
        "center_x": cx,
        "center_y": cy,
        "area": int(np.sum(mask)),
        "area_fraction": float(np.mean(mask)),
        "mean_B": float(np.mean(B[mask])),
        "mean_T": float(np.mean(T[mask])),
        "max_T": float(np.max(T[mask])),
        "mean_permeability": float(np.mean(P[mask])),
        "R_inside": float(np.mean(R[mask])),
        "X_inside": float(np.mean(X[mask])),
        "M_inside": float(np.mean(Mabs[mask])),
        "M_outside": float(np.mean(Mabs[outside])) if np.any(outside) else float(np.mean(Mabs[mask])),
        "temporal_concentration_delta": float(np.mean(Mabs[mask]) - (np.mean(Mabs[outside]) if np.any(outside) else np.mean(Mabs[mask]))),
        "chemical_potential_inside": float(np.mean(phi[mask])),
        "local_gradient_x": gx,
        "local_gradient_y": gy,
        "local_gradient_norm": float(math.sqrt(gx * gx + gy * gy)),
        "internal_circulation": float(np.mean(grad_mag(R + X + H + Mabs)[mask])),
    }


def init_tracks_from_preformation(comps, metrics, onset_step):
    tracks = {}
    active = set()
    for i, (comp, met) in enumerate(zip(comps, metrics)):
        cells = set(comp)
        onset_T = met["mean_T"]
        tr = PatchTrack(
            track_id=i,
            birth_step=onset_step,
            initial_cells=cells,
            current_cells=cells,
            onset_thickness=onset_T,
            onset_area_fraction=met["area_fraction"],
            onset_intermediate_band=int(INTERMEDIATE_LO <= onset_T < INTERMEDIATE_HI),
            steps=[], cx=[], cy=[], area_fraction=[], mean_T=[], mean_B=[],
            permeability=[], M_inside=[], M_outside=[], R_inside=[], X_inside=[],
            gradient_alignment=[], step_displacement=[], local_gradient_norm=[],
            internal_circulation=[],
        )
        tr.update(onset_step, cells, met)
        tracks[i] = tr
        active.add(i)
    return tracks, active, len(tracks)


def update_tracks_by_overlap(tracks, active_ids, next_id, step, comps, metrics):
    previous = {tid: tracks[tid].current_cells for tid in active_ids if tid in tracks}
    new_active = set()
    fusion = 0
    fission = 0
    used_prev = {}

    for comp, met in zip(comps, metrics):
        cells = set(comp)
        overlaps = []
        for tid, pcells in previous.items():
            ov = len(cells & pcells)
            if ov > 0:
                overlaps.append((tid, ov))
        if len(overlaps) > 1:
            fusion += 1
        if overlaps:
            tid = max(overlaps, key=lambda x: x[1])[0]
            used_prev[tid] = used_prev.get(tid, 0) + 1
        else:
            # New post-gradient patch; track but mark as non-preformed by onset thickness.
            tid = next_id
            next_id += 1
            onset_T = met["mean_T"]
            tracks[tid] = PatchTrack(
                track_id=tid,
                birth_step=step,
                initial_cells=cells,
                current_cells=cells,
                onset_thickness=onset_T,
                onset_area_fraction=met["area_fraction"],
                onset_intermediate_band=int(INTERMEDIATE_LO <= onset_T < INTERMEDIATE_HI),
                steps=[], cx=[], cy=[], area_fraction=[], mean_T=[], mean_B=[],
                permeability=[], M_inside=[], M_outside=[], R_inside=[], X_inside=[],
                gradient_alignment=[], step_displacement=[], local_gradient_norm=[],
                internal_circulation=[],
            )
        tracks[tid].update(step, cells, met)
        new_active.add(tid)

    fission = sum(1 for _, c in used_prev.items() if c > 1)
    rupture = len(active_ids - new_active)
    return tracks, new_active, next_id, fusion, fission, rupture


def physical_step(state, topo, src, has_temporal, has_lipid, shuffled, gradient_active, cfg, rng, step, budget):
    B, T, L, R, H, X, M, Mbuf = state["B"], state["T"], state["L"], state["R"], state["H"], state["X"], state["M"], state["Mbuf"]
    dt = cfg.dt
    residence, mineral, flow, u, v = topo["residence"], topo["mineral"], topo["flow_speed"], topo["u"], topo["v"]
    P = np.clip(np.exp(-2.7 * T) * (1.0 - 0.45 * B), 0.015, 1.0)

    D_R0, D_X0, D_H0, D_L0, D_M0 = 0.050, 0.060, 0.045, 0.032, 0.010
    D_R = D_R0 * np.clip(0.08 + 0.92 * P - 0.35 * residence, 0.02, 1.0)
    D_X = D_X0 * np.clip(0.10 + 0.90 * P - 0.30 * residence, 0.02, 1.0)
    D_H = D_H0 * np.clip(0.08 + 0.92 * P - 0.32 * residence, 0.02, 1.0)
    D_L = D_L0 * np.clip(0.06 + 0.94 * P - 0.45 * residence, 0.015, 1.0)
    D_M = D_M0 * np.clip(0.04 + 0.96 * P - 0.48 * residence, 0.01, 1.0)

    phi = 0.50 * R + 0.35 * L + 0.35 * H - 0.45 * X
    phix, phiy = grad(phi)

    # Chemical-potential transport only during gradient exposure.
    if gradient_active:
        chem_u = 0.085 * B * (0.25 + T) * P * phix
        chem_v = 0.085 * B * (0.25 + T) * P * phiy
    else:
        chem_u = np.zeros_like(B)
        chem_v = np.zeros_like(B)

    if has_temporal:
        delayed = Mbuf[(step - max(1, int(round(cfg.tau / dt)))) % len(Mbuf)]
        chem = R + 0.35 * H - X
        local_gamma = cfg.gamma * np.clip(1.0 - 0.34 * residence - 0.18 * B - 0.22 * T, 0.18, 1.2)
        local_beta = cfg.beta * (0.50 + 0.70 * residence + 0.16 * B + 0.10 * T)
        leak = 0.40 * P * M
        dM = (
            -local_gamma * M
            - local_beta * np.tanh(delayed)
            + 0.52 * chem * (0.35 + mineral)
            + variable_diffusion(M, D_M)
            + 0.06 * upwind_advection(M, u, v)
            + 0.20 * upwind_advection(M, chem_u, chem_v)
            - leak
        )
        M = np.clip(M + dt * dM + cfg.sigma * math.sqrt(dt) * rng.normal(size=M.shape), -6, 6)
        if shuffled and (step + 1) % 20 == 0:
            f = M.ravel().copy()
            rng.shuffle(f)
            M = f.reshape(M.shape)
        Mbuf[step % len(Mbuf)] = M
    else:
        M[:] = 0.0

    Mabs = np.abs(M)
    interface = np.clip(grad_mag(M) + 0.55 * grad_mag(R) + 0.30 * grad_mag(L) + 0.25 * grad_mag(H), 0, 2)
    circulation = grad_mag(R + X + H + Mabs)

    if has_lipid:
        basal = 0.003 * L**2 * (1 - B)
        interface_agg = 0.016 * L**2 * interface * (0.20 + mineral) * (1 - B)
        topo_nuc = 0.060 * sigmoid_array((L - 0.46) / 0.045) * sigmoid_array((residence - 0.52) / 0.055) * (0.25 + 0.75 * mineral) * (1 - B)
        lateral = 0.085 * neighbor_mean(B) * L * (0.25 + 0.75 * residence) * (1 + 0.25 * interface) * (1 - B)
        feedback = 0.035 * B * L * (0.30 + Mabs) * (0.30 + R + 0.20 * H) * (0.20 + residence) * (1 - B)
        fragmentation = B * (0.006 + 0.015 * flow + 0.012 * X)
        transport_B = upwind_advection(B, chem_u, chem_v)
        dB = basal + interface_agg + topo_nuc + lateral + feedback - fragmentation + transport_B

        fusion_T = 0.080 * B * neighbor_mean(B) * L * (0.35 + 0.65 * residence) * (1 - 0.25 * P)
        lateral_T = 0.055 * B**2 * neighbor_mean(T + B) * (0.20 + mineral)
        shear_T = 0.030 * (flow + 0.50 * np.sqrt(phix * phix + phiy * phiy)) * T
        stagnation = sigmoid_array((T - 0.75) / 0.08) * sigmoid_array((0.030 - circulation) / 0.010)
        rupture_T = 0.025 * stagnation * T
        turnover_T = 0.004 * T
        transport_T = upwind_advection(T, chem_u, chem_v)
        dT = fusion_T + lateral_T - shear_T - rupture_T - turnover_T + transport_T
    else:
        basal = interface_agg = topo_nuc = lateral = feedback = fragmentation = np.zeros_like(B)
        transport_B = transport_T = np.zeros_like(B)
        fusion_T = lateral_T = shear_T = rupture_T = turnover_T = np.zeros_like(B)
        dB = np.zeros_like(B)
        dT = np.zeros_like(T)

    B = np.clip(B + dt * dB, 0, 1)
    T = np.clip(T + dt * dT, 0, 2.5)
    T = np.minimum(T, 2.5 * B + 0.05)

    if has_lipid:
        lipid_consumption = 0.35 * (basal + interface_agg + topo_nuc + lateral + feedback) + 0.55 * (fusion_T + lateral_T)
        lipid_release = 0.10 * fragmentation + 0.12 * (shear_T + rupture_T + turnover_T)
        dL = variable_diffusion(L, D_L) + 0.04 * upwind_advection(L, u, v) + 0.0011 * src["L"] * (0.25 + 0.75 * P) - lipid_consumption + lipid_release - 0.0015 * L * flow
        L = np.clip(L + dt * dL, 0, 1.4)
    else:
        L[:] = 0.0

    dR = variable_diffusion(R, D_R) + 0.04 * upwind_advection(R, u, v) + 0.0013 * P * src["R"] - 0.008 * Mabs * R * (1 - 0.20 * B) - 0.0015 * R * flow
    dH = variable_diffusion(H, D_H) + 0.04 * upwind_advection(H, u, v) + 0.0008 * P * src["H"] - 0.002 * H * (0.2 + B) - 0.001 * H * flow
    dX = variable_diffusion(X, D_X) + 0.04 * upwind_advection(X, u, v) + 0.0045 * Mabs * (1 - 0.35 * B) - 0.012 * X * (0.35 + 0.65 * P) + 0.001 * flow + 0.0004 * src["X"]

    R = np.clip(R + dt * dR, 0, 1.8)
    H = np.clip(H + dt * dH, 0, 1.8)
    X = np.clip(X + dt * dX, 0, 1.8)

    for key, arr in [
        ("chemical_transport_B", np.abs(transport_B)),
        ("chemical_transport_T", np.abs(transport_T)),
        ("net_dB", dB),
        ("net_dT", dT),
        ("fusion_thickening", fusion_T),
        ("lateral_thickening", lateral_T),
        ("stagnation_rupture", rupture_T),
    ]:
        budget[key] = budget.get(key, 0.0) + float(np.mean(arr))

    state.update({"B": B, "T": T, "L": L, "R": R, "H": H, "X": X, "M": M, "Mbuf": Mbuf})
    diagnostics = {"P": P, "phi": phi, "phix": phix, "phiy": phiy, "circulation": circulation}
    return state, diagnostics


def classify_stage1_patch_status(patch_metrics: List[dict]) -> str:
    if len(patch_metrics) == 0:
        return "unformed"
    if any(INTERMEDIATE_LO <= p["mean_T"] < INTERMEDIATE_HI for p in patch_metrics):
        return "formed_intermediate"
    return "formed_nonintermediate"


def run_variant(rep: int, variant: str, cfg: Config, logger: ProgressLogger):
    rng = np.random.default_rng(cfg.seed + 1000 * rep + 97 * VARIANTS.index(variant))
    N = cfg.size
    flat = variant == "flat_temporal_lipid"
    has_temporal = "no_temporal" not in variant
    has_lipid = "no_lipid" not in variant
    shuffled = "shuffled" in variant

    topo = build_microtopography(N, rng, flat=flat)
    pre_src = source_fields(N, topo, gradient=False)
    grad_src = source_fields(N, topo, gradient=True)

    delay_steps = max(1, int(round(cfg.tau / cfg.dt)))
    M = rng.normal(0, 0.08, (N, N)) if has_temporal else np.zeros((N, N))
    Mbuf = np.zeros((delay_steps + 8, N, N))
    for k in range(delay_steps + 8):
        Mbuf[k] = M + rng.normal(0, 0.015, (N, N))

    state = {
        "B": np.zeros((N, N)),
        "T": np.zeros((N, N)),
        "L": 0.22 + 0.18 * pre_src["L"] + 0.10 * topo["pit_depth"] + 0.04 * rng.random((N, N)) if has_lipid else np.zeros((N, N)),
        "R": 0.28 + 0.15 * pre_src["R"] + 0.03 * rng.random((N, N)),
        "H": 0.20 + 0.12 * pre_src["H"] + 0.02 * rng.random((N, N)),
        "X": 0.03 + 0.08 * pre_src["X"] + 0.02 * rng.random((N, N)),
        "M": M,
        "Mbuf": Mbuf,
    }

    budget = {}
    timeseries_rows = []
    bulk_rows = []

    logger.log(f"preform rep={rep} variant={variant}")

    for step in range(cfg.preform_steps):
        state, diag = physical_step(state, topo, pre_src, has_temporal, has_lipid, shuffled, False, cfg, rng, step, budget)
        if step % cfg.sample_stride == 0 or step == cfg.preform_steps - 1:
            B, T = state["B"], state["T"]
            w = B + T
            bx, by = bulk_center(w)
            timeseries_rows.append({
                "replicate": rep,
                "variant": variant,
                "phase": "preformation",
                "step": step,
                "membrane_mean": float(np.mean(B)),
                "membrane_max": float(np.max(B)),
                "thickness_mean": float(np.mean(T)),
                "thickness_max": float(np.max(T)),
                "track_threshold_area": float(np.mean(B >= TRACK_THRESHOLD)),
                "bulk_center_x": bx,
                "bulk_center_y": by,
                "bulk_mass": float(np.sum(w)),
                "chemical_gradient_mean": float(np.mean(np.sqrt(diag["phix"]**2 + diag["phiy"]**2))),
            })

    # Stage-1 classification from observed patches.
    B, T, R, X, L, H, M = state["B"], state["T"], state["R"], state["X"], state["L"], state["H"], state["M"]
    P = np.clip(np.exp(-2.7 * T) * (1 - 0.45 * B), 0.015, 1.0)
    phi = 0.50 * R + 0.35 * L + 0.35 * H - 0.45 * X
    phix, phiy = grad(phi)

    comps = connected_components(B >= TRACK_THRESHOLD)
    patch_rows = []
    onset_metrics = []
    for cid, comp in enumerate(comps):
        mask = component_mask(B.shape, comp)
        met = compute_patch_metrics(mask, B, T, P, R, X, L, H, M, phi, phix, phiy)
        met.update({
            "replicate": rep,
            "variant": variant,
            "component_id": cid,
            "stage1_status_preliminary": "",
            "onset_intermediate_band": int(INTERMEDIATE_LO <= met["mean_T"] < INTERMEDIATE_HI),
        })
        patch_rows.append(met)
        onset_metrics.append(met)

    stage1_status = classify_stage1_patch_status(onset_metrics)
    for r in patch_rows:
        r["stage1_status"] = stage1_status

    status_row = {
        "replicate": rep,
        "variant": variant,
        "stage1_status": stage1_status,
        "has_membrane_patch": int(len(onset_metrics) > 0),
        "has_intermediate_patch": int(any(p["onset_intermediate_band"] for p in onset_metrics)),
        "patch_count": len(onset_metrics),
        "intermediate_patch_count": int(sum(p["onset_intermediate_band"] for p in onset_metrics)),
        "preform_membrane_mean": float(np.mean(B)),
        "preform_membrane_max": float(np.max(B)),
        "preform_thickness_mean": float(np.mean(T)),
        "preform_thickness_max": float(np.max(T)),
        "preform_track_area_fraction": float(np.mean(B >= TRACK_THRESHOLD)),
    }

    tracks, active_ids, next_id = init_tracks_from_preformation(comps, onset_metrics, cfg.preform_steps)

    logger.log(
        f"gradient rep={rep} variant={variant} stage1_status={stage1_status} "
        f"patches={len(onset_metrics)} intermediate={status_row['intermediate_patch_count']}"
    )

    # Initialize bulk movement at gradient onset for all models, including unformed.
    w0 = state["B"] + state["T"]
    bulk0_x, bulk0_y = bulk_center(w0)
    last_bulk_x, last_bulk_y = bulk0_x, bulk0_y
    bulk_path = 0.0

    fusion_total = fission_total = rupture_total = 0
    event_rows = []

    for gstep in range(cfg.gradient_steps):
        abs_step = cfg.preform_steps + gstep
        state, diag = physical_step(state, topo, grad_src, has_temporal, has_lipid, shuffled, True, cfg, rng, abs_step, budget)

        if gstep % cfg.sample_stride == 0 or gstep == cfg.gradient_steps - 1:
            B, T, R, X, L, H, M = state["B"], state["T"], state["R"], state["X"], state["L"], state["H"], state["M"]
            P = np.clip(np.exp(-2.7 * T) * (1 - 0.45 * B), 0.015, 1.0)
            phi = 0.50 * R + 0.35 * L + 0.35 * H - 0.45 * X
            phix, phiy = grad(phi)

            w = B + T
            bx, by = bulk_center(w)
            step_bulk_disp = math.sqrt((bx - last_bulk_x)**2 + (by - last_bulk_y)**2)
            bulk_path += step_bulk_disp
            last_bulk_x, last_bulk_y = bx, by
            bulk_rows.append({
                "replicate": rep,
                "variant": variant,
                "stage1_status": stage1_status,
                "step": abs_step,
                "bulk_center_x": bx,
                "bulk_center_y": by,
                "bulk_step_displacement": step_bulk_disp,
                "bulk_path_so_far": bulk_path,
                "bulk_net_displacement_from_onset": math.sqrt((bx - bulk0_x)**2 + (by - bulk0_y)**2),
                "bulk_mass": float(np.sum(w)),
                "membrane_mean": float(np.mean(B)),
                "thickness_mean": float(np.mean(T)),
            })

            comps = connected_components(B >= TRACK_THRESHOLD)
            valid = []
            mets = []
            for comp in comps:
                mask = component_mask(B.shape, comp)
                met = compute_patch_metrics(mask, B, T, P, R, X, L, H, M, phi, phix, phiy)
                if met:
                    valid.append(comp)
                    mets.append(met)
            tracks, active_ids, next_id, fu, fi, ru = update_tracks_by_overlap(tracks, active_ids, next_id, abs_step, valid, mets)
            fusion_total += fu
            fission_total += fi
            rupture_total += ru

            for tid in active_ids:
                tr = tracks[tid]
                if tr.steps and tr.steps[-1] == abs_step:
                    event_rows.append({
                        "replicate": rep,
                        "variant": variant,
                        "stage1_status": stage1_status,
                        "step": abs_step,
                        "track_id": tid,
                        "onset_intermediate_band": tr.onset_intermediate_band,
                        "center_x": tr.cx[-1],
                        "center_y": tr.cy[-1],
                        "step_displacement": tr.step_displacement[-1],
                        "gradient_alignment": tr.gradient_alignment[-1],
                        "mean_T": tr.mean_T[-1],
                        "mean_B": tr.mean_B[-1],
                        "mean_permeability": tr.permeability[-1],
                        "M_inside": tr.M_inside[-1],
                        "M_outside": tr.M_outside[-1],
                        "temporal_concentration_delta": tr.M_inside[-1] - tr.M_outside[-1],
                        "internal_circulation": tr.internal_circulation[-1],
                    })

            timeseries_rows.append({
                "replicate": rep,
                "variant": variant,
                "stage1_status": stage1_status,
                "phase": "gradient",
                "step": abs_step,
                "membrane_mean": float(np.mean(B)),
                "membrane_max": float(np.max(B)),
                "thickness_mean": float(np.mean(T)),
                "thickness_max": float(np.max(T)),
                "track_threshold_area": float(np.mean(B >= TRACK_THRESHOLD)),
                "active_track_count": len(active_ids),
                "active_intermediate_track_count": int(sum(tracks[t].onset_intermediate_band for t in active_ids if t in tracks)),
                "bulk_center_x": bx,
                "bulk_center_y": by,
                "bulk_mass": float(np.sum(w)),
                "chemical_gradient_mean": float(np.mean(np.sqrt(phix**2 + phiy**2))),
            })

    sample_dt = cfg.sample_stride * cfg.dt
    track_rows = []
    for tid, tr in tracks.items():
        if not tr.steps:
            continue
        chi_in, tc_in = autocorr_chi(tr.M_inside, sample_dt, cfg.gamma)
        chi_out, tc_out = autocorr_chi(tr.M_outside, sample_dt, cfg.gamma)
        net_disp = math.sqrt((tr.cx[-1] - tr.cx[0])**2 + (tr.cy[-1] - tr.cy[0])**2) if len(tr.cx) >= 2 else 0.0
        path_len = float(np.sum(tr.step_displacement))
        track_rows.append({
            "replicate": rep,
            "variant": variant,
            "stage1_status": stage1_status,
            "track_id": tid,
            "track_existed_at_gradient_onset": int(tr.birth_step == cfg.preform_steps),
            "onset_intermediate_band": tr.onset_intermediate_band,
            "onset_thickness": tr.onset_thickness,
            "onset_area_fraction": tr.onset_area_fraction,
            "n_observations": len(tr.steps),
            "lifetime_observed_steps": tr.steps[-1] - tr.steps[0] + cfg.sample_stride,
            "net_displacement": net_disp,
            "path_length": path_len,
            "persistence_ratio": net_disp / path_len if path_len > 1e-12 else 0.0,
            "mean_step_displacement": safe_mean(tr.step_displacement),
            "mean_gradient_alignment": safe_mean(tr.gradient_alignment),
            "mean_T": safe_mean(tr.mean_T),
            "max_T": max(tr.mean_T) if tr.mean_T else 0.0,
            "mean_B": safe_mean(tr.mean_B),
            "mean_permeability": safe_mean(tr.permeability),
            "temporal_concentration_delta_mean": safe_mean(np.asarray(tr.M_inside) - np.asarray(tr.M_outside)) if len(tr.M_inside) == len(tr.M_outside) else 0.0,
            "internal_circulation_mean": safe_mean(tr.internal_circulation),
            "chi_inside": chi_in,
            "chi_outside": chi_out,
            "chi_inside_minus_outside": chi_in - chi_out,
            "chi_inside_gt_1": int(chi_in > 1.0),
        })

    final_bulk_net = bulk_rows[-1]["bulk_net_displacement_from_onset"] if bulk_rows else 0.0
    final_bulk_path = bulk_rows[-1]["bulk_path_so_far"] if bulk_rows else 0.0

    final_row = {
        "replicate": rep,
        "variant": variant,
        "stage1_status": stage1_status,
        "patch_count_at_gradient_onset": status_row["patch_count"],
        "intermediate_patch_count_at_gradient_onset": status_row["intermediate_patch_count"],
        "bulk_net_displacement": final_bulk_net,
        "bulk_path_length": final_bulk_path,
        "bulk_persistence_ratio": final_bulk_net / final_bulk_path if final_bulk_path > 1e-12 else 0.0,
        "preformed_track_net_displacement_mean": safe_mean([r["net_displacement"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1]),
        "preformed_track_path_length_mean": safe_mean([r["path_length"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1]),
        "preformed_track_chi_inside_mean": safe_mean([r["chi_inside"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1]),
        "preformed_track_chi_inside_gt_1_mean": safe_mean([r["chi_inside_gt_1"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1]),
        "intermediate_track_net_displacement_mean": safe_mean([r["net_displacement"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1 and r["onset_intermediate_band"] == 1]),
        "intermediate_track_path_length_mean": safe_mean([r["path_length"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1 and r["onset_intermediate_band"] == 1]),
        "intermediate_track_chi_inside_mean": safe_mean([r["chi_inside"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1 and r["onset_intermediate_band"] == 1]),
        "intermediate_track_chi_inside_gt_1_mean": safe_mean([r["chi_inside_gt_1"] for r in track_rows if r["track_existed_at_gradient_onset"] == 1 and r["onset_intermediate_band"] == 1]),
        "fusion_count_total": fusion_total,
        "fission_count_total": fission_total,
        "rupture_count_total": rupture_total,
    }

    budget_rows = [
        {
            "replicate": rep,
            "variant": variant,
            "stage1_status": stage1_status,
            "term": k,
            "time_integrated_mean": v,
            "mean_per_step": v / max(1, cfg.preform_steps + cfg.gradient_steps),
        }
        for k, v in budget.items()
    ]

    if cfg.save_fields and rep == 0:
        fdir = cfg.outdir / "field_arrays"
        mkdir(fdir)
        np.savez_compressed(
            fdir / f"final_fields_{variant}.npz",
            B=state["B"], T=state["T"], L=state["L"], R=state["R"], H=state["H"], X=state["X"], M=state["M"],
            pit_depth=topo["pit_depth"], residence=topo["residence"], mineral=topo["mineral"],
        )

    return {
        "preformation_status": pd.DataFrame([status_row]),
        "preformation_patch_metrics": pd.DataFrame(patch_rows),
        "preformed_patch_motility_tracks": pd.DataFrame(track_rows),
        "patch_motility_event_timeseries": pd.DataFrame(event_rows),
        "bulk_material_motility": pd.DataFrame(bulk_rows),
        "gradient_timeseries_metrics": pd.DataFrame(timeseries_rows),
        "final_condition_metrics": pd.DataFrame([final_row]),
        "motility_budget": pd.DataFrame(budget_rows),
    }


def write_rep_outputs(rep: int, outdir: Path, outputs: Dict[str, pd.DataFrame]) -> None:
    repdir = outdir / "rep_outputs"
    mkdir(repdir)
    for name, df in outputs.items():
        df.to_csv(repdir / f"{name}_rep{rep:03d}.csv", index=False)
    (repdir / f"rep{rep:03d}.done.json").write_text(json.dumps({"replicate": rep, "completed": True}, indent=2), encoding="utf-8")


def read_rep_outputs(outdir: Path, name: str) -> pd.DataFrame:
    repdir = outdir / "rep_outputs"
    frames = []
    for p in sorted(repdir.glob(f"{name}_rep*.csv")):
        if p.exists() and p.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or "onset_thickness" not in df.columns:
        return df
    out = df.copy()
    out["onset_thickness_bin"] = pd.cut(out["onset_thickness"], bins=THICKNESS_BINS, labels=THICKNESS_LABELS, include_lowest=True)
    return out


def combine_and_report(cfg: Config, logger: ProgressLogger):
    names = [
        "preformation_status",
        "preformation_patch_metrics",
        "preformed_patch_motility_tracks",
        "patch_motility_event_timeseries",
        "bulk_material_motility",
        "gradient_timeseries_metrics",
        "final_condition_metrics",
        "motility_budget",
    ]
    data = {n: read_rep_outputs(cfg.outdir, n) for n in names}
    data["preformed_patch_motility_tracks"] = add_bins(data["preformed_patch_motility_tracks"])

    for n, df in data.items():
        df.to_csv(cfg.outdir / f"{n}.csv", index=False)

    status_summary = summarize(
        data["preformation_status"], ["variant", "stage1_status"],
        ["has_membrane_patch", "has_intermediate_patch", "patch_count", "intermediate_patch_count",
         "preform_membrane_mean", "preform_membrane_max", "preform_thickness_mean", "preform_thickness_max",
         "preform_track_area_fraction"],
    )
    status_summary.to_csv(cfg.outdir / "preformation_status_summary.csv", index=False)

    variant_summary = summarize(
        data["final_condition_metrics"], ["variant", "stage1_status"],
        ["bulk_net_displacement", "bulk_path_length", "bulk_persistence_ratio",
         "preformed_track_net_displacement_mean", "preformed_track_path_length_mean",
         "preformed_track_chi_inside_mean", "preformed_track_chi_inside_gt_1_mean",
         "intermediate_track_net_displacement_mean", "intermediate_track_path_length_mean",
         "intermediate_track_chi_inside_mean", "intermediate_track_chi_inside_gt_1_mean",
         "fusion_count_total", "fission_count_total", "rupture_count_total"],
    )
    variant_summary.to_csv(cfg.outdir / "motility_summary_by_variant.csv", index=False)

    status_motility_summary = summarize(
        data["final_condition_metrics"], ["stage1_status"],
        ["bulk_net_displacement", "bulk_path_length", "bulk_persistence_ratio",
         "preformed_track_net_displacement_mean", "preformed_track_path_length_mean",
         "preformed_track_chi_inside_mean", "preformed_track_chi_inside_gt_1_mean",
         "intermediate_track_net_displacement_mean", "intermediate_track_path_length_mean",
         "intermediate_track_chi_inside_mean", "intermediate_track_chi_inside_gt_1_mean"],
    )
    status_motility_summary.to_csv(cfg.outdir / "motility_summary_by_stage1_status.csv", index=False)

    preformed_only = data["preformed_patch_motility_tracks"]
    if len(preformed_only) and "track_existed_at_gradient_onset" in preformed_only.columns:
        preformed_only = preformed_only[preformed_only["track_existed_at_gradient_onset"] == 1]
    preformed_summary = summarize(
        preformed_only, ["variant", "stage1_status"],
        ["onset_intermediate_band", "onset_thickness", "net_displacement", "path_length", "persistence_ratio",
         "mean_step_displacement", "mean_gradient_alignment", "mean_T", "mean_permeability",
         "temporal_concentration_delta_mean", "internal_circulation_mean", "chi_inside",
         "chi_outside", "chi_inside_minus_outside", "chi_inside_gt_1"],
    )
    preformed_summary.to_csv(cfg.outdir / "preformed_patch_motility_summary.csv", index=False)

    bin_summary = summarize(
        preformed_only, ["variant", "stage1_status", "onset_thickness_bin"],
        ["net_displacement", "path_length", "persistence_ratio",
         "mean_gradient_alignment", "mean_T", "mean_permeability",
         "temporal_concentration_delta_mean", "internal_circulation_mean",
         "chi_inside", "chi_outside", "chi_inside_gt_1"],
    )
    bin_summary.to_csv(cfg.outdir / "motility_summary_by_onset_thickness_bin.csv", index=False)

    budget_summary = summarize(
        data["motility_budget"], ["variant", "stage1_status", "term"],
        ["time_integrated_mean", "mean_per_step"],
    )
    budget_summary.to_csv(cfg.outdir / "motility_budget_summary.csv", index=False)

    if cfg.make_figures:
        make_figures(cfg.outdir, data["final_condition_metrics"], preformed_only)

    write_report(cfg, status_summary, variant_summary, status_motility_summary, preformed_summary, bin_summary, cfg.outdir)

    manifest = {
        "script": "stage2_compare_formed_unformed_gradient_motility.py",
        "mode": cfg.mode,
        "seed": cfg.seed,
        "replicates": cfg.replicates,
        "size": cfg.size,
        "preform_steps": cfg.preform_steps,
        "gradient_steps": cfg.gradient_steps,
        "track_threshold": TRACK_THRESHOLD,
        "intermediate_thickness_band": [INTERMEDIATE_LO, INTERMEDIATE_HI],
        "variants": VARIANTS,
        "hard_exclusions": [
            "goal", "reward", "reinforcement_learning", "optimizer", "weight_update",
            "survival_predictor", "set_point", "homeostatic_objective",
            "exploration_instruction", "move_to_resource_instruction",
            "target_gradient_alignment", "target_thickness", "target_chi", "success_condition",
        ],
    }
    (cfg.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = cfg.outdir / "stage2_compare_formed_unformed_gradient_motility_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in cfg.outdir.rglob("*"):
            if p.is_file() and p.name != zip_path.name:
                z.write(p, arcname=str(p.relative_to(cfg.outdir)))
    logger.log(f"wrote {zip_path}")


def write_report(cfg, status_summary, variant_summary, status_motility_summary, preformed_summary, bin_summary, outdir: Path):
    lines = []
    lines.append("Stage 2: Gradient Motility of Stage-1 Formed vs Unformed Models")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Design")
    lines.append("------")
    lines.append("Each model is first run through the Stage-1 membrane-thickness self-selection dynamics. The resulting state is classified as unformed, formed_nonintermediate, or formed_intermediate. Only then is the model exposed to a chemical-gradient world. Movement is measured both for preformed patches and for the bulk membrane-material center of mass.")
    lines.append("")
    lines.append("Configuration")
    lines.append("-------------")
    lines.append(f"mode: {cfg.mode}")
    lines.append(f"replicates: {cfg.replicates}")
    lines.append(f"size: {cfg.size}")
    lines.append(f"preform_steps: {cfg.preform_steps}")
    lines.append(f"gradient_steps: {cfg.gradient_steps}")
    lines.append(f"track_threshold: {TRACK_THRESHOLD}")
    lines.append(f"intermediate_thickness_band: {INTERMEDIATE_LO} <= mean_T < {INTERMEDIATE_HI}")
    lines.append("")
    lines.append("Preformation status summary")
    lines.append("---------------------------")
    lines.append(status_summary.to_string(index=False) if len(status_summary) else "No status summary.")
    lines.append("")
    lines.append("Motility summary by variant")
    lines.append("---------------------------")
    lines.append(variant_summary.to_string(index=False) if len(variant_summary) else "No variant summary.")
    lines.append("")
    lines.append("Motility summary by Stage-1 status")
    lines.append("----------------------------------")
    lines.append(status_motility_summary.to_string(index=False) if len(status_motility_summary) else "No status motility summary.")
    lines.append("")
    lines.append("Preformed patch motility summary")
    lines.append("--------------------------------")
    lines.append(preformed_summary.to_string(index=False) if len(preformed_summary) else "No preformed patch tracks.")
    lines.append("")
    lines.append("Motility summary by onset thickness bin")
    lines.append("--------------------------------------")
    lines.append(bin_summary.to_string(index=False) if len(bin_summary) else "No thickness-bin summary.")
    lines.append("")
    lines.append("Fact-use note")
    lines.append("-------------")
    lines.append("This report is observational. It does not define success by displacement, gradient alignment, chi > 1, or biological labels.")
    (outdir / "stage2_formed_unformed_integrated_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(outdir: Path, final_df: pd.DataFrame, tracks: pd.DataFrame):
    if plt is None:
        return
    figdir = outdir / "figures"
    mkdir(figdir)

    if len(final_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        for status, g in final_df.groupby("stage1_status"):
            ax.scatter(g["bulk_net_displacement"], g["bulk_path_length"], label=status, alpha=0.8)
        ax.set_xlabel("Bulk membrane-material net displacement")
        ax.set_ylabel("Bulk membrane-material path length")
        ax.set_title("Formed vs unformed: bulk material motility")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / "Figure_1_bulk_formed_vs_unformed.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    if len(tracks):
        fig, ax = plt.subplots(figsize=(8, 5))
        for status, g in tracks.groupby("stage1_status"):
            ax.scatter(g["net_displacement"], g["chi_inside"], label=status, alpha=0.65, s=20)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Preformed patch net displacement")
        ax.set_ylabel("Preformed patch chi_inside")
        ax.set_title("Preformed patch motility and chi")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / "Figure_2_patch_displacement_chi.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def run(cfg: Config):
    mkdir(cfg.outdir)
    logger = ProgressLogger(cfg.outdir)
    logger.log("Stage 2 formed-vs-unformed gradient motility started")
    logger.log(f"mode={cfg.mode} reps={cfg.replicates} size={cfg.size} preform={cfg.preform_steps} gradient={cfg.gradient_steps}")

    repdir = cfg.outdir / "rep_outputs"
    mkdir(repdir)

    for rep in range(cfg.replicates):
        done = repdir / f"rep{rep:03d}.done.json"
        if cfg.resume and done.exists():
            logger.log(f"SKIP completed replicate {rep}")
            continue
        logger.log(f"START replicate {rep + 1}/{cfg.replicates}")
        combined = {}
        for variant in VARIANTS:
            out = run_variant(rep, variant, cfg, logger)
            for name, df in out.items():
                combined.setdefault(name, []).append(df)
        combined_df = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() for name, frames in combined.items()}
        write_rep_outputs(rep, cfg.outdir, combined_df)
        logger.log(f"DONE replicate {rep + 1}/{cfg.replicates}")

    combine_and_report(cfg, logger)
    (cfg.outdir / "_RUN_DONE.json").write_text(json.dumps({"completed": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    logger.log("completed")


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--preform-steps", type=int, default=None)
    ap.add_argument("--gradient-steps", type=int, default=None)
    ap.add_argument("--sample-stride", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.08)
    args = ap.parse_args()

    if args.mode == "smoke":
        reps, size, preform, gradient, stride = 1, 30, 1200, 300, 5
    elif args.mode == "quick":
        reps, size, preform, gradient, stride = 6, 52, 1200, 900, 5
    else:
        reps, size, preform, gradient, stride = 16, 72, 3600, 2200, 5

    if args.replicates is not None:
        reps = args.replicates
    if args.size is not None:
        size = args.size
    if args.preform_steps is not None:
        preform = args.preform_steps
    if args.gradient_steps is not None:
        gradient = args.gradient_steps
    if args.sample_stride is not None:
        stride = args.sample_stride

    return Config(
        mode=args.mode,
        outdir=Path(args.outdir).expanduser(),
        seed=args.seed,
        replicates=reps,
        size=size,
        preform_steps=preform,
        gradient_steps=gradient,
        sample_stride=stride,
        resume=args.resume,
        make_figures=not args.no_figures,
        save_fields=args.save_fields,
        dt=args.dt,
        gamma=args.gamma,
        beta=args.beta,
        tau=args.tau,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    run(parse_args())
