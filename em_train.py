#!/usr/bin/env python3
"""
EarthMind UNet3D Training
=========================
Diffusion-based downscaling model trained on GraphCast → AORC pairs.
GraphCast fields are pre-materialized per IC to minimise repeated I/O,
and an optional patch cap (``--max-patches-per-ic``) allows shorter runs
for debugging or resource-constrained environments.

Usage
-----
    # Full run (all patches):
    python em_train.py

    # Capped run (500 patches per IC, single-process data loading):
    python em_train.py --max-patches-per-ic 500 --num-workers 0
"""

__author__      = "Somnath Luitel and Manmeet Singh"
__lab__         = ("AI Research Lab, Department of Earth Sciences, "
                   "Environmental and Atmospheric Sciences, "
                   "Western Kentucky University")
__publication__ = "Earthmind-Highres (2026)"
__version__     = "1.0.0"
__license__     = "<License>"

import os
import gc
import time
import argparse
import resource

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet3DConditionModel, LCMScheduler


# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(
    description="EarthMind UNet3D training",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--max-patches-per-ic", type=int, default=0,
                    help="Max non-NaN patches per IC (0 = all)")
parser.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers for AORC loading")
parser.add_argument("--prefetch-factor", type=int, default=2,
                    help="DataLoader prefetch factor (ignored when num-workers=0)")
args = parser.parse_args()

MAX_PATCHES    = args.max_patches_per_ic
NUM_WORKERS    = args.num_workers
PREFETCH_FACTOR = args.prefetch_factor


# =============================================================================
# Helpers
# =============================================================================
def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log_mem(prefix: str) -> None:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)
    msg = f"[{ts()}] [MEM] {prefix} | CPU maxrss={rss_gb:.2f} GB"
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / (1024.0 ** 3)
        reserv = torch.cuda.memory_reserved()  / (1024.0 ** 3)
        msg   += f" | GPU alloc={alloc:.2f} GB reserved={reserv:.2f} GB"
    print(msg)


# =============================================================================
# Paths
# =============================================================================
MODEL_DIR = "/media/airlab/ROCSTOR/earthmind_highres/models_unet3d_fast"
DONE_FILE = os.path.join(MODEL_DIR, "done_ics.txt")
os.makedirs(MODEL_DIR, exist_ok=True)

done_ics: set = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        done_ics = {ln.strip() for ln in f if ln.strip()}
print(f"[{ts()}] Already-completed ICs: {len(done_ics)}")


# =============================================================================
# Data loading (lazy)
# =============================================================================
AORC_CHUNKS = {"time": 24, "latitude": 256, "longitude": 256}
GC_CHUNKS   = {"time": 1, "prediction_timedelta": 12,
               "latitude": 256, "longitude": 256}

ds_aorc = xr.open_mfdataset(
    "/media/airlab/ROCSTOR/earthmind_highres/noaa_aorc_usa/"
    "noaa_aorc_usa_????_day_????????.nc",
    combine="by_coords",
    chunks=AORC_CHUNKS,
)
print(f"[{ts()}] AORC loaded (lazy)\n{ds_aorc}")

ds_gc = xr.open_zarr(
    "/home/airlab/Documents/airlab/earthmind_highres/"
    "graphcast_training_2020_2021_new.zarr",
    chunks=GC_CHUNKS,
).sel(time=slice("2021-01-27", "2021-12-31"))
print(f"[{ts()}] GraphCast loaded (lazy)\n{ds_gc}")

gc_min   = xr.open_zarr("/home/airlab/Documents/airlab/earthmind_highres/graphcast_training_2020_2021_min.zarr").compute()
gc_max   = xr.open_zarr("/home/airlab/Documents/airlab/earthmind_highres/graphcast_training_2020_2021_max.zarr").compute()
aorc_min = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/usa_aorc_min.nc").compute()
aorc_max = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/usa_aorc_max.nc").compute()

ds_svf  = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/SKY_VIEW_FACTOR_1k_c230606.nc")
ds_topo = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/elevation.nc")


# =============================================================================
# Normalisation
# =============================================================================
PRECIP_GC   = {"total_precipitation_6hr"}
PRECIP_AORC = {"APCP_surface"}


def minmax_norm(da: xr.DataArray, vmin, vmax, eps: float = 1e-12) -> xr.DataArray:
    denom = xr.where(np.abs(vmax - vmin) < eps, np.nan, vmax - vmin)
    return (da - vmin) / denom


def log_minmax_norm(da: xr.DataArray, vmin, vmax, eps: float = 1e-12) -> xr.DataArray:
    da_l   = xr.ufuncs.log1p(da.clip(min=0))
    vmin_l = xr.ufuncs.log1p(vmin.clip(min=0))
    vmax_l = xr.ufuncs.log1p(vmax.clip(min=0))
    denom  = xr.where(np.abs(vmax_l - vmin_l) < eps, np.nan, vmax_l - vmin_l)
    return (da_l - vmin_l) / denom


def _normalise_dataset(ds: xr.Dataset, ds_min, ds_max,
                        precip_vars: set) -> xr.Dataset:
    out = ds.copy()
    for v in ds.data_vars:
        if v not in ds_min.data_vars or v not in ds_max.data_vars:
            continue
        fn  = log_minmax_norm if v in precip_vars else minmax_norm
        out[v] = fn(ds[v], ds_min[v], ds_max[v])
    return out.map(lambda da: da.clip(0, 1) if isinstance(da, xr.DataArray) else da)


# =============================================================================
# GraphCast pre-processing: expand pressure levels, normalise, regrid
# =============================================================================
LEVELS     = [850, 700, 500]
level_vars = [v for v, da in ds_gc.data_vars.items() if "level" in da.dims]

gc_expanded = ds_gc.copy()
for v in level_vars:
    for lev in LEVELS:
        gc_expanded[f"{v}_level_{lev}"] = ds_gc[v].sel(level=lev).drop_vars("level")
gc_expanded = gc_expanded.drop_vars(level_vars)
print(f"[{ts()}] Pressure-level variables expanded")

gc_norm = _normalise_dataset(gc_expanded, gc_min, gc_max, PRECIP_GC)

gc_ll = (gc_norm
         .assign_coords(lon=(((gc_norm.lon + 180) % 360) - 180))
         .sortby("lon")
         .rename({"lat": "latitude", "lon": "longitude"}))

gc_interp = gc_ll.interp(
    latitude=ds_aorc.latitude,
    longitude=ds_aorc.longitude,
    method="linear",
)
print(f"[{ts()}] GraphCast interpolated to AORC grid (lazy)")

# Attach static fields (topo + sky-view factor)
for ds_static, old_name, new_name in [
    (ds_topo, {"lat": "latitude", "lon": "longitude"}, "norm_elevation"),
    (ds_svf,  {"lat": "latitude", "lon": "longitude"}, "SKY_VIEW_FACTOR"),
]:
    ds_r = ds_static.rename(old_name).interp(
        latitude=ds_aorc.latitude,
        longitude=ds_aorc.longitude,
        method="linear",
    ).fillna(0)
    varname = "topo" if new_name == "norm_elevation" else "svf"
    gc_interp[varname] = ds_r[new_name]

aorc_norm = _normalise_dataset(ds_aorc, aorc_min, aorc_max, PRECIP_AORC)
print(f"[{ts()}] AORC normalised (lazy)")


# =============================================================================
# Solar zenith angle (NOAA algorithm)
# =============================================================================
def cos_sza_noaa(valid_time: np.ndarray,
                  lat_deg:   np.ndarray,
                  lon_deg:   np.ndarray) -> np.ndarray:
    dt  = valid_time.astype("datetime64[ns]")
    doy = (dt.astype("datetime64[D]") - dt.astype("datetime64[Y]")).astype(np.int64) + 1
    sec = (dt - dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64)
    fhr = sec / 3600.0

    lat   = np.deg2rad(lat_deg)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (fhr - 12.0) / 24.0)

    eot = 229.18 * (
          0.000075
        + 0.001868 * np.cos(gamma)   - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2*gamma) - 0.040849 * np.sin(2*gamma)
    )
    decl = (
          0.006918
        - 0.399912 * np.cos(gamma)   + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2*gamma) + 0.000907 * np.sin(2*gamma)
        - 0.002697 * np.cos(3*gamma) + 0.001480 * np.sin(3*gamma)
    )
    ha      = np.deg2rad(fhr * 60.0 / 4.0 + eot / 4.0 + lon_deg - 180.0)
    cos_zen = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha)
    return np.clip(cos_zen, 0.0, 1.0).astype(np.float32)


def cos_sza_patch(valid_time_1d: np.ndarray,
                   lat_1d:       np.ndarray,
                   lon_1d:       np.ndarray) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(lat_1d.astype(np.float32),
                                lon_1d.astype(np.float32), indexing="ij")
    return cos_sza_noaa(
        valid_time_1d[:, None, None].astype("datetime64[ns]"),
        lat2d[None], lon2d[None],
    )


# =============================================================================
# Feature lists
# =============================================================================
GC_3D_VARS = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "mean_sea_level_pressure", "total_precipitation_6hr",
    "geopotential_level_850",      "geopotential_level_700",      "geopotential_level_500",
    "specific_humidity_level_850", "specific_humidity_level_700", "specific_humidity_level_500",
    "temperature_level_850",       "temperature_level_700",       "temperature_level_500",
    "vertical_velocity_level_850", "vertical_velocity_level_700", "vertical_velocity_level_500",
]
GC_2D_VARS = ["topo", "svf"]

AORC_VARS = [
    "APCP_surface", "DLWRF_surface", "DSWRF_surface", "PRES_surface",
    "SPFH_2maboveground", "TMP_2maboveground",
    "UGRD_10maboveground", "VGRD_10maboveground",
]


# =============================================================================
# Model, scheduler, optimiser
# =============================================================================
PATCH      = 64
T_COND     = 12
T_TARGET   = 67
BATCH_SIZE = 1

COND_CH   = 20   # 17 (3-D GC) + 2 (static 2-D) + 1 (cos-SZA)
TARGET_CH =  8   # 8 AORC variables

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{ts()}] Device: {device}")

model = UNet3DConditionModel(
    sample_size=None,
    in_channels=TARGET_CH + COND_CH,
    out_channels=TARGET_CH,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types  =("UpBlock3D",   "UpBlock3D",   "UpBlock3D",   "UpBlock3D"),
    norm_num_groups=8,
    cross_attention_dim=1,
    attention_head_dim=8,
).to(device)

noise_scheduler = LCMScheduler(num_train_timesteps=1000)
loss_fn         = nn.MSELoss()
optimiser       = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
encoder_hidden  = torch.zeros(1, 1, 1, device=device)


# =============================================================================
# Checkpoint resume
# =============================================================================
BEST_INFO_PATH = os.path.join(MODEL_DIR, "global_best.txt")
global_best_loss = float("inf")

if os.path.exists(BEST_INFO_PATH):
    best_ckpt = None
    with open(BEST_INFO_PATH) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            if k == "best_loss_global": global_best_loss = float(v)
            if k == "checkpoint":       best_ckpt = v

    if best_ckpt and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimizer_state_dict"])
        if "best_loss_global" in ckpt:
            global_best_loss = float(ckpt["best_loss_global"])
        print(f"[{ts()}] Resumed from global best  loss={global_best_loss}  ckpt={best_ckpt}")
    else:
        print(f"[{ts()}] global_best.txt present but checkpoint not found — starting fresh")
else:
    print(f"[{ts()}] No checkpoint found — starting fresh")


# =============================================================================
# Patch index loading + bounding-box computation
# =============================================================================
patch_file = os.path.join(MODEL_DIR, f"common_valid_patches_patch{PATCH}.npy")
if not os.path.exists(patch_file):
    raise FileNotFoundError(f"Patch index file not found: {patch_file}")

valid_patches = np.load(patch_file, allow_pickle=True).tolist()
print(f"[{ts()}] Loaded {len(valid_patches)} valid patch indices from {patch_file}")


def _patch_bbox(patches: list, ps: int) -> tuple[int, int, int, int]:
    ii = [p[0] for p in patches]; jj = [p[1] for p in patches]
    return min(ii), max(ii) + ps, min(jj), max(jj) + ps


BBOX_I0, BBOX_I1, BBOX_J0, BBOX_J1 = _patch_bbox(valid_patches, PATCH)
est_gb = (len(GC_3D_VARS) * T_COND + len(GC_2D_VARS)) * \
         (BBOX_I1-BBOX_I0) * (BBOX_J1-BBOX_J0) * 4 / (1024**3)
print(f"[{ts()}] GC bbox i=[{BBOX_I0},{BBOX_I1}) j=[{BBOX_J0},{BBOX_J1}) "
      f"~{est_gb:.1f} GB per IC")

log_mem("startup")
print(f"[{ts()}] max_patches_per_ic={'all' if MAX_PATCHES == 0 else MAX_PATCHES}  "
      f"num_workers={NUM_WORKERS}")


# =============================================================================
# GraphCast pre-materialisation (once per IC)
# =============================================================================
def prematerialize_gc(gc_ic_lazy: xr.Dataset) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """
    Load the spatial bounding box of the GraphCast IC into memory as plain
    NumPy arrays.  Called once per IC; subsequent patch slices are O(1).
    """
    t0      = time.time()
    gc_bbox = gc_ic_lazy.isel(
        latitude =slice(BBOX_I0, BBOX_I1),
        longitude=slice(BBOX_J0, BBOX_J1),
    )
    lat_np = gc_bbox.latitude.values
    lon_np = gc_bbox.longitude.values

    gc_3d: dict[str, np.ndarray] = {}
    for v in GC_3D_VARS:
        arr = gc_bbox[v].transpose("prediction_timedelta", "latitude", "longitude").values
        gc_3d[v] = arr.astype(np.float32)

    gc_2d: dict[str, np.ndarray] = {}
    for v in GC_2D_VARS:
        arr = gc_bbox[v].transpose("latitude", "longitude").values
        gc_2d[v] = arr.astype(np.float32)

    print(f"[{ts()}]   GC pre-materialised in {time.time()-t0:.1f}s")
    return gc_3d, gc_2d, lat_np, lon_np


# =============================================================================
# Training loop
# =============================================================================
model.train()

ic_times   = gc_interp.time.values
global_step = 0
print(f"[{ts()}] IC times: {len(ic_times)}  ({ic_times[0]} … {ic_times[-1]})")

for time_ic in ic_times:
    ic_str = str(np.datetime64(time_ic, "ns"))

    if ic_str in done_ics:
        print(f"[{ts()}] SKIP (done) {ic_str}")
        continue

    print(f"\n[{ts()}] === IC: {ic_str} ===")

    gc_ic       = gc_interp.sel(time=time_ic)
    aorc_window = aorc_norm.sel(time=slice(time_ic, time_ic + np.timedelta64(66, "h")))

    if aorc_window.sizes["time"] != T_TARGET:
        print(f"[{ts()}]   SKIP — AORC window length {aorc_window.sizes['time']} ≠ {T_TARGET}")
        continue

    valid_time_1d = (
        np.datetime64(time_ic, "ns") + gc_ic["prediction_timedelta"].values
    ).astype("datetime64[ns]")

    # Deterministic per-IC shuffle
    seed = int(
        np.uint64(np.datetime64(time_ic, "ns").astype("int64"))
        ^ np.uint64(0x9E3779B97F4A7C15)
    )
    rng             = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
    patches_shuffled = valid_patches.copy()
    rng.shuffle(patches_shuffled)

    print(f"[{ts()}]   Pre-materialising GraphCast bbox …")
    gc_3d, gc_2d, lat_bbox, lon_bbox = prematerialize_gc(gc_ic)
    log_mem(f"after GC prematerialise {ic_str}")

    t_ic_start    = time.time()
    t_log_start   = time.time()
    trained_this_ic = 0

    for (i0, j0) in patches_shuffled:

        # ── Conditioning patch (GraphCast) ────────────────────────────────
        gi0, gi1 = i0 - BBOX_I0, i0 - BBOX_I0 + PATCH
        gj0, gj1 = j0 - BBOX_J0, j0 - BBOX_J0 + PATCH

        lat_1d = lat_bbox[gi0:gi1]
        lon_1d = lon_bbox[gj0:gj1]
        cos_sza = cos_sza_patch(valid_time_1d, lat_1d, lon_1d).astype(np.float32)

        X_list = [gc_3d[v][:, gi0:gi1, gj0:gj1] for v in GC_3D_VARS]
        for v in GC_2D_VARS:
            tile = np.broadcast_to(gc_2d[v][gi0:gi1, gj0:gj1][None],
                                   (T_COND, PATCH, PATCH))
            X_list.append(tile)
        X_list.append(cos_sza)
        X_np = np.stack(X_list, axis=0)   # (C_cond, T_cond, P, P)

        # ── Target patch (AORC) ───────────────────────────────────────────
        aorc_patch = aorc_window.isel(
            latitude =slice(i0, i0 + PATCH),
            longitude=slice(j0, j0 + PATCH),
        )
        Y_np = np.stack(
            [aorc_patch[v].transpose("time", "latitude", "longitude")
                          .values.astype(np.float32)
             for v in AORC_VARS],
            axis=0,
        )   # (C_target, T_target, P, P)

        if np.isnan(X_np).any() or np.isnan(Y_np).any():
            print(f"[{ts()}]   NaN — skipping patch ({i0},{j0})")
            del X_np, Y_np, X_list, cos_sza
            gc.collect()
            continue

        # ── Diffusion training step ───────────────────────────────────────
        x = torch.from_numpy(X_np)[None].to(device, non_blocking=True)  # (1, C_cond,   T_cond,   P, P)
        y = torch.from_numpy(Y_np)[None].to(device, non_blocking=True)  # (1, C_target, T_target, P, P)

        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (BATCH_SIZE,), device=device,
        ).long()
        noise   = torch.randn_like(y)
        noisy_y = noise_scheduler.add_noise(y, noise, timesteps)

        x_up    = F.interpolate(x, size=(T_TARGET, PATCH, PATCH),
                                mode="trilinear", align_corners=False)
        net_in  = torch.cat([noisy_y, x_up], dim=1)

        noise_pred = model(
            sample=net_in,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden,
        ).sample

        loss = loss_fn(noise_pred, noise)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        loss_val = float(loss.detach().item())

        # ── Save global best ──────────────────────────────────────────────
        if loss_val < global_best_loss:
            global_best_loss = loss_val
            ic_tag  = ic_str.replace(":", "").replace("-", "")[:15]
            ckpt_path = os.path.join(MODEL_DIR, f"unet3d_best_{ic_tag}.pt")
            torch.save(
                {"ic": ic_str, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimiser.state_dict(),
                 "best_loss_global": global_best_loss},
                ckpt_path,
            )
            with open(BEST_INFO_PATH, "w") as f:
                f.write(f"best_loss_global={global_best_loss}\n"
                        f"ic={ic_str}\n"
                        f"checkpoint={ckpt_path}\n")
            print(f"[{ts()}]   NEW BEST loss={global_best_loss:.6f}  "
                  f"patch=({i0},{j0})  ckpt={ckpt_path}")

        global_step    += 1
        trained_this_ic += 1

        if global_step % 10 == 0:
            elapsed = time.time() - t_log_start
            print(f"[{ts()}]   step={global_step} "
                  f"loss={loss_val:.6f} "
                  f"{elapsed:.1f}s/10steps ({10/elapsed:.2f} sps) "
                  f"trained={trained_this_ic}"
                  + (f"/{MAX_PATCHES}" if MAX_PATCHES > 0 else ""))
            log_mem(f"step {global_step}")
            t_log_start = time.time()

        # ── Cleanup ───────────────────────────────────────────────────────
        del X_np, Y_np, X_list, cos_sza, x, y
        del noise, noisy_y, x_up, net_in, noise_pred, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if MAX_PATCHES > 0 and trained_this_ic >= MAX_PATCHES:
            print(f"[{ts()}]   Patch cap reached ({MAX_PATCHES}) — next IC")
            break

    # ── IC done ───────────────────────────────────────────────────────────
    del gc_3d, gc_2d, lat_bbox, lon_bbox
    gc.collect()

    ic_elapsed = time.time() - t_ic_start
    with open(DONE_FILE, "a") as f:
        f.write(ic_str + "\n")
    done_ics.add(ic_str)

    print(f"[{ts()}] DONE IC {ic_str} | "
          f"patches={trained_this_ic} | "
          f"time={ic_elapsed/60:.1f} min | "
          f"best_loss={global_best_loss:.6f}")
    log_mem(f"done IC {ic_str}")