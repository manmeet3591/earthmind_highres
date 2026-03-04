import os
import gc
import resource
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from diffusers import UNet3DConditionModel, LCMScheduler

# -----------------------------
# Paths / bookkeeping
# -----------------------------
done_file = "/scratch/08105/ms86336/earthmind_highres/done_ics.txt"
model_dir = "/scratch/08105/ms86336/earthmind_highres/models_unet3d"
os.makedirs(model_dir, exist_ok=True)

done_ics = set()
if os.path.exists(done_file):
    with open(done_file, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                done_ics.add(s)
print("already done ICs:", len(done_ics))

global_best_info = os.path.join(model_dir, "global_best.txt")

# -----------------------------
# Memory logging (CPU + GPU)
# -----------------------------
def log_mem(prefix: str):
    # ru_maxrss is KB on Linux
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    msg = f"[MEM] {prefix} | CPU maxrss={rss_gb:.2f} GB"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024.0**3)
        reserv = torch.cuda.memory_reserved() / (1024.0**3)
        msg += f" | GPU alloc={alloc:.2f} GB reserved={reserv:.2f} GB"
    print(msg)

# -----------------------------
# Chunking knobs (IMPORTANT)
# -----------------------------
# Adjust if needed. Goal: avoid gigantic compute / task graphs.
AORC_CHUNKS = {"time": 24, "latitude": 256, "longitude": 256}
GC_CHUNKS   = {"time": 1, "prediction_timedelta": 12, "latitude": 256, "longitude": 256}

# -----------------------------
# AORC (huge) - keep lazy/dask with chunks
# -----------------------------
ds_aorc = xr.open_mfdataset(
    "/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_????_day_????????.nc",
    combine="by_coords",
    chunks=AORC_CHUNKS
)
print(ds_aorc)

# -----------------------------
# GraphCast zarr (chunked)
# -----------------------------
ds_gc = xr.open_zarr(
    "/scratch/08105/ms86336/earthmind_highres/graphcast_training_2020_2021.zarr",
    chunks=GC_CHUNKS
).sel(time=slice("2021-04-03", "2021-12-31")) # This is where we have to change
print(ds_gc)

# min/max (small) - compute once
gc_min = xr.open_zarr("/scratch/08105/ms86336/earthmind_highres/graphcast_training_2020_2021_min.zarr").compute()
gc_max = xr.open_zarr("/scratch/08105/ms86336/earthmind_highres/graphcast_training_2020_2021_max.zarr").compute()

# AORC min/max (small) - compute once
aorc_min = xr.open_dataset("/scratch/08105/ms86336/earthmind_highres/usa_aorc_min.nc").compute()
aorc_max = xr.open_dataset("/scratch/08105/ms86336/earthmind_highres/usa_aorc_max.nc").compute()

# static fields
ds_svf = xr.open_dataset("/scratch/08105/ms86336/earthmind_highres/SKY_VIEW_FACTOR_1k_c230606.nc")
ds_topo = xr.open_dataset("/scratch/08105/ms86336/earthmind_highres/elevation.nc")

# -----------------------------
# 1) Expand GraphCast level variables into level{850,700,500}
# -----------------------------
levels = [850, 700, 500]
level_vars = [v for v, da in ds_gc.data_vars.items() if "level" in da.dims]

out = ds_gc.copy()
for v in level_vars:
    for lev in levels:
        out[f"{v}_level_{lev}"] = ds_gc[v].sel(level=lev).drop_vars("level")
out = out.drop_vars(level_vars)
print(out)

# -----------------------------
# 2) Normalize GraphCast (min/max; log-minmax for precip)
# -----------------------------
def minmax_norm(da, vmin, vmax, eps=1e-12):
    denom = (vmax - vmin)
    denom = xr.where(np.abs(denom) < eps, np.nan, denom)
    return (da - vmin) / denom

def log_minmax_norm_precip(da, vmin, vmax, eps=1e-12):
    da0 = da.clip(min=0)
    vmin0 = vmin.clip(min=0)
    vmax0 = vmax.clip(min=0)
    da_l   = xr.ufuncs.log1p(da0)
    vmin_l = xr.ufuncs.log1p(vmin0)
    vmax_l = xr.ufuncs.log1p(vmax0)
    denom = (vmax_l - vmin_l)
    denom = xr.where(np.abs(denom) < eps, np.nan, denom)
    return (da_l - vmin_l) / denom

precip_gc = {"total_precipitation_6hr"}

out_norm = out.copy()
for v in out.data_vars:
    if v not in gc_min.data_vars or v not in gc_max.data_vars:
        continue
    vmin = gc_min[v]
    vmax = gc_max[v]
    if v in precip_gc:
        out_norm[v] = log_minmax_norm_precip(out[v], vmin, vmax)
    else:
        out_norm[v] = minmax_norm(out[v], vmin, vmax)

out_norm = out_norm.map(lambda da: da.clip(0, 1) if isinstance(da, xr.DataArray) else da)

# -----------------------------
# 3) Lon to [-180,180], rename, interp to AORC grid
# -----------------------------
out_ll = out_norm.assign_coords(lon=(((out_norm.lon + 180) % 360) - 180)).sortby("lon")
out_ll = out_ll.rename({"lat": "latitude", "lon": "longitude"})

out_norm_interp = out_ll.interp(
    latitude=ds_aorc.latitude,
    longitude=ds_aorc.longitude,
    method="linear"
)
print(out_norm_interp)

# -----------------------------
# 4) Interpolate topo + svf to AORC grid and attach
# -----------------------------
ds_topo = ds_topo.rename({"lat": "latitude", "lon": "longitude"})
ds_svf  = ds_svf.rename({"lat": "latitude", "lon": "longitude"})

ds_topo_interp = ds_topo.interp(
    latitude=ds_aorc.latitude,
    longitude=ds_aorc.longitude,
    method="linear"
).fillna(0)

ds_svf_interp = ds_svf.interp(
    latitude=ds_aorc.latitude,
    longitude=ds_aorc.longitude,
    method="linear"
)

out_norm_interp["topo"] = ds_topo_interp["norm_elevation"]
out_norm_interp["svf"]  = ds_svf_interp["SKY_VIEW_FACTOR"]

# -----------------------------
# 5) Normalize AORC (min/max; log-minmax for APCP_surface)
# -----------------------------
precip_aorc = {"APCP_surface"}

ds_aorc_norm = ds_aorc.copy()
for v in ds_aorc.data_vars:
    if v not in aorc_min.data_vars or v not in aorc_max.data_vars:
        continue
    vmin = aorc_min[v]
    vmax = aorc_max[v]
    if v in precip_aorc:
        ds_aorc_norm[v] = log_minmax_norm_precip(ds_aorc[v], vmin, vmax)
    else:
        ds_aorc_norm[v] = minmax_norm(ds_aorc[v], vmin, vmax)

ds_aorc_norm = ds_aorc_norm.map(lambda da: da.clip(0, 1) if isinstance(da, xr.DataArray) else da)
print(ds_aorc_norm)

# -----------------------------
# 6) Solar zenith (NOAA method) - patch-level
# -----------------------------
def cos_sza_noaa_3d(valid_time_np, lat_deg_np, lon_deg_np):
    dt = valid_time_np.astype("datetime64[ns]")
    doy = (dt.astype("datetime64[D]") - dt.astype("datetime64[Y]")).astype(np.int64) + 1
    sec_of_day = (dt - dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64)
    frac_hour = sec_of_day / 3600.0

    lat = np.deg2rad(lat_deg_np)
    gamma = 2.0*np.pi/365.0 * (doy - 1 + (frac_hour - 12.0)/24.0)

    eot = 229.18 * (
        0.000075
        + 0.001868*np.cos(gamma)
        - 0.032077*np.sin(gamma)
        - 0.014615*np.cos(2*gamma)
        - 0.040849*np.sin(2*gamma)
    )

    decl = (
        0.006918
        - 0.399912*np.cos(gamma)
        + 0.070257*np.sin(gamma)
        - 0.006758*np.cos(2*gamma)
        + 0.000907*np.sin(2*gamma)
        - 0.002697*np.cos(3*gamma)
        + 0.00148*np.sin(2*gamma)
    )

    tst = frac_hour*60.0 + eot + 4.0*lon_deg_np
    ha = np.deg2rad(tst/4.0 - 180.0)

    cos_zen = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha)
    return np.clip(cos_zen, 0.0, 1.0).astype("float32")

def cos_sza_patch(valid_time_1d, lat_1d, lon_1d):
    """
    valid_time_1d: (Tcond,) datetime64
    lat_1d: (patch,) float
    lon_1d: (patch,) float
    returns: (Tcond, patch, patch) float32
    """
    vt = valid_time_1d.astype("datetime64[ns]")
    lat2d, lon2d = np.meshgrid(lat_1d.astype(np.float32), lon_1d.astype(np.float32), indexing="ij")
    vt3d = vt[:, None, None]  # (T,1,1)
    return cos_sza_noaa_3d(vt3d, lat2d[None, :, :], lon2d[None, :, :])  # (T,P,P)

# -----------------------------
# 7) Define feature lists
# -----------------------------
vars_keep_3d_no_sza = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "geopotential_level_850",
    "geopotential_level_700",
    "geopotential_level_500",
    "specific_humidity_level_850",
    "specific_humidity_level_700",
    "specific_humidity_level_500",
    "temperature_level_850",
    "temperature_level_700",
    "temperature_level_500",
    "vertical_velocity_level_850",
    "vertical_velocity_level_700",
    "vertical_velocity_level_500",
]
vars_keep_2d = ["topo", "svf"]

aorc_vars_keep = [
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]

# -----------------------------
# 8) Build model + scheduler + optimizer
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

cond_channels = 20   # 17 (3d) + 2 (2d) + 1 (cos_sza)
target_channels = 8  # AORC has 8 vars
cross_attention_dim = 1

model = UNet3DConditionModel(
    sample_size=None,
    in_channels=target_channels + cond_channels,
    out_channels=target_channels,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock3D","DownBlock3D","DownBlock3D","DownBlock3D"),
    up_block_types=("UpBlock3D","UpBlock3D","UpBlock3D","UpBlock3D"),
    norm_num_groups=8,
    cross_attention_dim=cross_attention_dim,
    attention_head_dim=8,
).to(device)

noise_scheduler = LCMScheduler(num_train_timesteps=1000)

loss_fn = nn.MSELoss()
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

encoder_hidden_states = torch.zeros(1, 1, cross_attention_dim, device=device)  # (B=1, seq=1, dim=1)

# -----------------------------
# 9) Training loop: ICs then patches
# -----------------------------
model.train()

patch = 64
Tcond = 12
Ttarget = 67
B = 1

times = out_norm_interp.time.values
print("num IC times:", len(times), " first:", times[0], " last:", times[-1])

step = 0
global_best_loss = float("inf")

# Resume global best if exists
if os.path.exists(global_best_info):
    best_ckpt_path = None
    with open(global_best_info, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("best_loss_global="):
                global_best_loss = float(line.split("=", 1)[1])
            if line.startswith("checkpoint="):
                best_ckpt_path = line.split("=", 1)[1]

    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "best_loss_global" in ckpt:
            global_best_loss = float(ckpt["best_loss_global"])
        print("LOADED GLOBAL BEST:", global_best_loss, "| from:", best_ckpt_path)
    else:
        print("global_best.txt exists but checkpoint path missing/not found")
else:
    print("No global best found; starting fresh")

# ---- load precomputed common patches ----
common_patch_file = os.path.join(model_dir, f"common_valid_patches_patch{patch}.npy")
if not os.path.exists(common_patch_file):
    raise FileNotFoundError(f"Missing common patch file: {common_patch_file}")
valid_patches = np.load(common_patch_file, allow_pickle=True).tolist()
print("Loaded common valid patches:", len(valid_patches), "from:", common_patch_file)

log_mem("startup")

for time_ic in times:
    ic_str = str(np.datetime64(time_ic, "ns"))

    if ic_str in done_ics:
        print("\n=== SKIP (already done) IC:", ic_str, "===")
        continue

    print("\n=== IC:", ic_str, "===")

    # Select GraphCast IC (lazy)
    gc_ic = out_norm_interp.sel(time=time_ic)

    # Select AORC target window (lazy)
    ds_aorc_win = ds_aorc_norm.sel(time=slice(time_ic, time_ic + np.timedelta64(66, "h")))
    if ds_aorc_win.sizes["time"] != Ttarget:
        print("skip (AORC window length)", ds_aorc_win.sizes["time"])
        continue

    print(f"IC {ic_str}: valid patches = {len(valid_patches)}")
    if len(valid_patches) == 0:
        print("skip IC (no valid patches)")
        continue

    # Precompute valid_time_1d once per IC for sza (tiny)
    valid_time_1d = (np.datetime64(time_ic, "ns") + gc_ic["prediction_timedelta"].values).astype("datetime64[ns]")

    # -------------------------------------------------------------------
    # IMPORTANT CHANGE: randomize patch iteration order (NOT sequential)
    # Deterministic per-IC so resumes are reproducible
    # -------------------------------------------------------------------
    seed = (np.uint64(np.datetime64(time_ic, "ns").astype("int64")) ^ np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(int(seed))
    patches_shuffled = valid_patches.copy()
    rng.shuffle(patches_shuffled)
    # -------------------------------------------------------------------

    for (i0, j0) in patches_shuffled:
        # ----------------------------
        # Build X patch (GraphCast cond) WITHOUT full-grid cos_sza
        # ----------------------------
        gc_patch = gc_ic.isel(latitude=slice(i0, i0+patch), longitude=slice(j0, j0+patch))

        # Pull patch coordinates (tiny)
        lat_1d = gc_patch.latitude.values
        lon_1d = gc_patch.longitude.values

        # Compute cos_sza for patch only: (Tcond, patch, patch)
        cos_sza_np = cos_sza_patch(valid_time_1d, lat_1d, lon_1d).astype(np.float32)

        # Load the 3D vars for patch into numpy
        X_list = []
        for v in vars_keep_3d_no_sza:
            arr = gc_patch[v].transpose("prediction_timedelta", "latitude", "longitude").values  # (Tcond,P,P)
            X_list.append(arr.astype(np.float32))

        # 2D vars expanded across time
        for v in vars_keep_2d:
            arr2 = gc_patch[v].transpose("latitude", "longitude").values.astype(np.float32)  # (P,P)
            arr3 = np.broadcast_to(arr2[None, :, :], (Tcond, patch, patch))
            X_list.append(arr3)

        # Append cos_sza as last channel
        X_list.append(cos_sza_np)

        # Stack to (C=20, Tcond, P, P)
        X_np = np.stack(X_list, axis=0)  # (20, 12, P, P)

        # ----------------------------
        # Build Y patch (AORC target)
        # ----------------------------
        aorc_patch = ds_aorc_win.isel(latitude=slice(i0, i0+patch), longitude=slice(j0, j0+patch))

        Y_list = []
        for v in aorc_vars_keep:
            arr = aorc_patch[v].transpose("time", "latitude", "longitude").data
            arr = np.asarray(arr).astype(np.float32)  # (Ttarget,P,P)
            Y_list.append(arr)
        Y_np = np.stack(Y_list, axis=0)  # (8, 67, P, P)

        # Safety NaN check
        if np.isnan(X_np).any() or np.isnan(Y_np).any():
            print("UNEXPECTED NaN | IC", ic_str, "| patch", i0, j0)
            del X_np, Y_np, X_list, Y_list, cos_sza_np
            gc.collect()
            continue

        # ----------------------------
        # Torch tensors
        # ----------------------------
        x = torch.from_numpy(X_np)[None].to(device, non_blocking=True)  # (1,20,12,P,P)
        y = torch.from_numpy(Y_np)[None].to(device, non_blocking=True)  # (1,8,67,P,P)

        # ----------------------------
        # One diffusion training step
        # ----------------------------
        noise = torch.randn_like(y)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy_y = noise_scheduler.add_noise(y, noise, timesteps)

        x_match = F.interpolate(x, size=(Ttarget, patch, patch), mode="trilinear", align_corners=False)
        net_input = torch.cat([noisy_y, x_match], dim=1)

        noise_pred = model(
            sample=net_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        loss_val = loss_fn(noise_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss_val.backward()
        opt.step()

        loss_float = float(loss_val.detach().item())

        # ---------- global best ----------
        if loss_float < global_best_loss:
            global_best_loss = loss_float
            ic_safe = ic_str.replace(":", "").replace("-", "").replace(".000000000", "")
            if "T" in ic_safe and len(ic_safe) > 15:
                ic_safe = ic_safe[:15]

            global_best_path = os.path.join(model_dir, f"unet3d_global_best_ic_{ic_safe}.pt")
            torch.save(
                {
                    "ic": ic_str,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_loss_global": global_best_loss,
                },
                global_best_path
            )
            with open(global_best_info, "w") as f:
                f.write(f"best_loss_global={global_best_loss}\n")
                f.write(f"ic={ic_str}\n")
                f.write(f"checkpoint={global_best_path}\n")

            print("NEW GLOBAL BEST:", global_best_loss, "| saved:", global_best_path,
                  "| IC", ic_str, "| patch", i0, j0)

        step += 1
        if step % 10 == 0:
            print("step", step, "IC", str(time_ic), "patch", i0, j0, "loss", loss_float)
            log_mem(f"step {step}")

        # Cleanup
        del X_np, Y_np, X_list, Y_list, cos_sza_np
        del x, y, noise, noisy_y, x_match, net_input, noise_pred, loss_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Mark IC done after finishing all patches
    with open(done_file, "a") as f:
        f.write(ic_str + "\n")
    done_ics.add(ic_str)
    print("DONE IC:", ic_str, "best loss:", global_best_loss)
    log_mem(f"done IC {ic_str}")
