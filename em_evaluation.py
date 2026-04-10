#!/usr/bin/env python3
"""
EarthMind CONUS Evaluation Script
==================================
Evaluates EarthMind downscaling output against AORC (ground truth),
HRRR (NWP baseline), and GraphCast (AI baseline) over CONUS.

Spectral analysis uses 2D radially-averaged PSD with Hann windowing,
appropriate for regional (non-periodic) domains.
Reference: Harris et al. (2022, JAMES), Leinonen et al. (2020).
"""
from __future__ import annotations

__author__      = "Somnath Luitel and Manmeet Singh"
__lab__         = ("AI Research Lab, Department of Earth Sciences, "
                   "Environmental and Atmospheric Sciences, "
                   "Western Kentucky University")
__publication__ = "Earthmind-Highres (2026)"
__version__     = "1.0.0"
__license__     = "<License>"

import argparse
import csv
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import xarray as xr
import zarr
import fsspec

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker

from scipy.interpolate import griddata
from pyproj import CRS, Transformer

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn("cartopy not installed — maps will lack coastlines/borders")

# =============================================================================
# Nature-quality figure constants  (two-column layout: maps | PSD+scatter)
# =============================================================================
FIGURE_DPI        = 400

# ---- Font: Arial/Helvetica for Nature compliance
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",          # non-italic math text
    "pdf.fonttype":     42,                 # TrueType in PDF (Nature requirement)
    "ps.fonttype":      42,
})

# ---- Font sizes (readable in LaTeX at 100% zoom, 14-in figure) --------------
FONT_SUPTITLE      = 18.0
FONT_PANEL_TITLE   = 18.0
FONT_CBAR_LABEL    = 10.0
FONT_CBAR_TICK     = 15.0
FONT_SCATTER_AXIS  = 15.0
FONT_SCATTER_TICK  = 15.0
FONT_SCATTER_TITLE = 18.0
FONT_STATS_BOX     = 12.5
FONT_PSD_AXIS      = 15.0
FONT_PSD_TICK      = 15.0
FONT_PSD_TITLE     = 15.0
FONT_PSD_LEGEND    = 13.0
FONT_GEO_LABEL     = 14.0     # gridline lat/lon labels on maps

# ---- Colorbar sizing
CBAR_FRACTION      = 0.035
CBAR_PAD           = 0.015
CBAR_SHRINK        = 0.85
CBAR_N_TICKS       = 4

# ---- Scatter styling
SCATTER_S          = 4.0
SCATTER_ALPHA      = 0.15
SCATTER_COLOR      = "steelblue"

# ---- PSD plot styling
PSD_COLORS = {
    "AORC":      "#1b9e77",   # teal-green
    "GraphCast": "#d95f02",   # orange
    "HRRR":      "#7570b3",   # purple
    "Earthmind": "#e7298a",   # magenta-pink
}
PSD_LINEWIDTHS = {
    "AORC":      2.2,
    "GraphCast": 1.6,
    "HRRR":      1.6,
    "Earthmind": 2.0,
}
PSD_LINESTYLES = {
    "AORC":      "-",
    "GraphCast": "--",
    "HRRR":      "-.",
    "Earthmind": "-",
}

# =============================================================================
# AORC / model constants
# =============================================================================
AORC_S3_BASE = "s3://noaa-nws-aorc-v1-1-1km"

AORC_VARS = [
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]

VAR_UNITS = {
    "APCP_surface":        "mm/hr",
    "DLWRF_surface":       "W/m²",
    "DSWRF_surface":       "W/m²",
    "PRES_surface":        "Pa",
    "SPFH_2maboveground":  "kg/kg",
    "TMP_2maboveground":   "K",
    "UGRD_10maboveground": "m/s",
    "VGRD_10maboveground": "m/s",
}

VAR_LONG_NAMES = {
    "APCP_surface":        "Precipitation",
    "DLWRF_surface":       "Downward LW Radiation",
    "DSWRF_surface":       "Downward SW Radiation",
    "PRES_surface":        "Surface Pressure",
    "SPFH_2maboveground":  "Specific Humidity (2 m)",
    "TMP_2maboveground":   "Temperature (2 m)",
    "UGRD_10maboveground": "U-Wind (10 m)",
    "VGRD_10maboveground": "V-Wind (10 m)",
}

HRRR_MAP = {
    "APCP_surface":        ("surface",          "PRATE"),
    "DLWRF_surface":       ("surface",          "DLWRF"),
    "DSWRF_surface":       ("surface",          "DSWRF"),
    "PRES_surface":        ("surface",          "PRES"),
    "TMP_2maboveground":   ("2m_above_ground",  "TMP"),
    "SPFH_2maboveground":  ("2m_above_ground",  "SPFH"),
    "UGRD_10maboveground": ("10m_above_ground", "UGRD"),
    "VGRD_10maboveground": ("10m_above_ground", "VGRD"),
}

GC_MAP = {
    "TMP_2maboveground":   "2m_temperature",
    "UGRD_10maboveground": "10m_u_component_of_wind",
    "VGRD_10maboveground": "10m_v_component_of_wind",
    "APCP_surface_6hr":    "total_precipitation_6hr",
}

VARS_NO_GC = {
    "SPFH_2maboveground",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
}

# Variables that are non-negative by physical definition.
NONNEG_VARS = {
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "SPFH_2maboveground",
}

HRRR_MAX_FHR = 48
PRECIP_CMAP  = "YlGnBu"
STATE_CMAP   = "RdYlBu_r"
QQ_N_QUANT   = 2000

# Earth radius in km for wavelength computation
EARTH_RADIUS_KM = 6371.0

# =============================================================================
# Logging / utilities
# =============================================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(p) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def normalize_lon_180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64).copy()
    return np.where(lon > 180.0, lon - 360.0, lon)

def to_dt64ns(x) -> np.datetime64:
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[ns]")
    x_arr = np.asarray(x)
    if x_arr.ndim > 0:
        x_arr = np.asarray(x_arr.flat[0])
    if np.issubdtype(x_arr.dtype, np.integer):
        return np.datetime64(int(x_arr), "ns")
    if isinstance(x, (bytes, bytearray)):
        x = x.decode()
    return np.datetime64(str(x)).astype("datetime64[ns]")

def fmt_time(t) -> str:
    return str(np.datetime64(t, "m"))

def _clip_nonneg(arr: np.ndarray, var: str) -> np.ndarray:
    if var in NONNEG_VARS:
        return np.clip(arr, 0.0, None)
    return arr

# =============================================================================
# 2D Radially-Averaged Power Spectral Density (with Hann window)
# =============================================================================
def compute_radial_psd(field_2d: np.ndarray,
                        lat: np.ndarray,
                        lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = field_2d.shape

    field = field_2d.copy().astype(np.float64)
    mask_finite = np.isfinite(field)
    if mask_finite.sum() < 100:
        return np.array([]), np.array([])
    field_mean = np.nanmean(field)
    field[~mask_finite] = field_mean

    field -= field_mean

    win_y = np.hanning(ny)
    win_x = np.hanning(nx)
    window = np.outer(win_y, win_x)
    field_windowed = field * window
    window_power = np.mean(window ** 2)

    fft2 = np.fft.fft2(field_windowed)
    fft2_shifted = np.fft.fftshift(fft2)
    power = (np.abs(fft2_shifted) ** 2) / (nx * ny) / window_power

    mid_lat = np.mean(lat)
    dlat_km = np.abs(np.mean(np.diff(lat))) * (np.pi / 180.0) * EARTH_RADIUS_KM
    dlon_km = (np.abs(np.mean(np.diff(lon))) * (np.pi / 180.0)
                * EARTH_RADIUS_KM * np.cos(np.deg2rad(mid_lat)))

    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dlat_km))
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dlon_km))
    freq_x_2d, freq_y_2d = np.meshgrid(freq_x, freq_y)
    freq_r = np.sqrt(freq_x_2d ** 2 + freq_y_2d ** 2)

    max_freq = min(
        np.max(np.abs(freq_y[freq_y > 0])),
        np.max(np.abs(freq_x[freq_x > 0]))
    )
    n_bins = min(nx, ny) // 2
    freq_bins = np.linspace(0, max_freq, n_bins + 1)
    freq_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])

    psd_radial = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        ring = (freq_r >= freq_bins[i]) & (freq_r < freq_bins[i + 1])
        if ring.sum() > 0:
            psd_radial[i] = np.mean(power[ring])

    valid = (freq_centers > 0) & (psd_radial > 0)
    freq_valid = freq_centers[valid]
    psd_valid = psd_radial[valid]

    if freq_valid.size == 0:
        return np.array([]), np.array([])

    wavelength_km = 1.0 / freq_valid
    return wavelength_km, psd_valid

# =============================================================================
# Q-Q bias correction
# =============================================================================
def apply_qq_correction(em_2d: np.ndarray,
                         aorc_2d: np.ndarray) -> np.ndarray:
    em_flat = em_2d.ravel().astype(np.float64)
    ao_flat = aorc_2d.ravel().astype(np.float64)
    paired  = np.isfinite(em_flat) & np.isfinite(ao_flat)
    if paired.sum() < 200:
        log("  [QQ] < 200 paired pixels -- correction skipped")
        return em_2d.copy().astype(np.float32)
    pcts = np.linspace(0.0, 100.0, QQ_N_QUANT)
    em_q = np.percentile(em_flat[paired], pcts)
    ao_q = np.percentile(ao_flat[paired], pcts)
    corrected      = em_flat.copy()
    fin            = np.isfinite(em_flat)
    corrected[fin] = np.interp(em_flat[fin], em_q, ao_q)
    return corrected.reshape(em_2d.shape).astype(np.float32)

# =============================================================================
# AORC land-mask propagation
# =============================================================================
def _apply_mask(aorc_arr: np.ndarray, *arrays) -> tuple:
    mask = np.isfinite(aorc_arr)
    return tuple(
        np.where(mask, a, np.nan) if a is not None else None
        for a in arrays)

# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(model_2d: np.ndarray, truth_2d: np.ndarray) -> dict:
    x = np.asarray(model_2d).ravel()
    y = np.asarray(truth_2d).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 50:
        return dict(N=n, r=np.nan, rmse=np.nan, bias=np.nan,
                    mae=np.nan, std_err=np.nan)
    err   = x - y
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    bias  = float(np.mean(err))
    mae   = float(np.mean(np.abs(err)))
    std_e = float(np.std(err))
    sx, sy = float(np.std(x)), float(np.std(y))
    r = float(np.corrcoef(x, y)[0, 1]) if (sx > 0 and sy > 0) else np.nan
    return dict(N=n, r=r, rmse=rmse, bias=bias, mae=mae, std_err=std_e)

def robust_minmax(arr, p_lo=2, p_hi=98):
    a = np.asarray(arr); a = a[np.isfinite(a)]
    if a.size == 0: return -1.0, 1.0
    lo = float(np.percentile(a, p_lo))
    hi = float(np.percentile(a, p_hi))
    if lo == hi: lo -= 1.0; hi += 1.0
    return lo, hi

# =============================================================================
# Scatter panel helper
# =============================================================================
def _draw_scatter(ax, model2d, aorc2d, title, sample_max=300_000):
    if model2d is None or aorc2d is None:
        ax.set_title(f"{title}\n(missing)", fontsize=FONT_SCATTER_TITLE)
        ax.axis("off"); return False
    x = np.asarray(aorc2d).ravel()
    y = np.asarray(model2d).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 50:
        ax.set_title(f"{title}\n(N<50)", fontsize=FONT_SCATTER_TITLE)
        ax.axis("off"); return False
    if x.size > sample_max:
        rng = np.random.default_rng(42)
        idx = rng.choice(x.size, size=sample_max, replace=False)
        x, y = x[idx], y[idx]
    lo = min(float(np.percentile(x, 0.5)),  float(np.percentile(y, 0.5)))
    hi = max(float(np.percentile(x, 99.5)), float(np.percentile(y, 99.5)))
    if lo == hi: lo -= 1.0; hi += 1.0
    met = compute_metrics(y, x)
    ax.scatter(x, y, s=SCATTER_S, alpha=SCATTER_ALPHA, c=SCATTER_COLOR,
               edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("AORC", fontsize=FONT_SCATTER_AXIS)
    ax.set_ylabel("Model", fontsize=FONT_SCATTER_AXIS)
    ax.set_title(title, fontsize=FONT_SCATTER_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONT_SCATTER_TICK)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.grid(True, linewidth=0.3, alpha=0.35, color="gray", linestyle="--")
    ax.set_axisbelow(True)

    r_s = f"{met['r']:.4f}" if np.isfinite(met["r"]) else "NaN"
    txt = (f"N={met['N']:,}\n"
           f"r={r_s}\n"
           f"RMSE={met['rmse']:.4g}\n"
           f"Bias={met['bias']:.4g}\n"
           f"MAE={met['mae']:.4g}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=FONT_STATS_BOX, family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#aaaaaa", alpha=0.88, linewidth=0.5))
    return True

# =============================================================================
# PSD panel helper
# =============================================================================
def _draw_psd(ax, psd_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
              var_name: str, unit: str) -> bool:
    drawn = False
    for label in ["AORC", "GraphCast", "HRRR", "Earthmind"]:
        if label not in psd_data:
            continue
        wl, psd = psd_data[label]
        if wl.size == 0:
            continue
        ax.loglog(
            wl, psd,
            color=PSD_COLORS.get(label, "gray"),
            linewidth=PSD_LINEWIDTHS.get(label, 1.5),
            linestyle=PSD_LINESTYLES.get(label, "-"),
            label=label,
            alpha=0.9,
        )
        drawn = True

    if not drawn:
        ax.set_title("PSD (no data)", fontsize=FONT_PSD_TITLE)
        ax.axis("off")
        return False

    ax.set_xlabel("Wavelength (km)", fontsize=FONT_PSD_AXIS)
    ax.set_ylabel("Power Spectral Density", fontsize=FONT_PSD_AXIS)
    ax.set_title(f"Radial PSD — {var_name}", fontsize=FONT_PSD_TITLE,
                 fontweight="bold")
    ax.tick_params(labelsize=FONT_PSD_TICK)
    ax.invert_xaxis()
    ax.grid(True, which="both", linewidth=0.3, alpha=0.35,
            color="gray", linestyle="--")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.legend(fontsize=FONT_PSD_LEGEND, loc="upper right",
              frameon=True, fancybox=True, framealpha=0.9,
              edgecolor="#aaaaaa")
    return True

# =============================================================================
# Geo axes helper
# =============================================================================
def _add_geo_ax(fig, spec, extent=None):
    if HAS_CARTOPY:
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", linewidth=0.55, color="k")
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="#555")
        ax.add_feature(cfeature.STATES,  linewidth=0.22, edgecolor="#888")
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2,
                          color="gray", alpha=0.40, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlocator     = mticker.MaxNLocator(nbins=5)
        gl.ylocator     = mticker.MaxNLocator(nbins=5)
        gl.xlabel_style = {"size": FONT_GEO_LABEL}
        gl.ylabel_style = {"size": FONT_GEO_LABEL}
    else:
        ax = fig.add_subplot(spec)
        ax.tick_params(labelsize=FONT_GEO_LABEL)
    return ax

# =============================================================================
# Combined figure  (two-column: maps left | PSD + scatter right)
# =============================================================================
def save_combined_figure(
        spatial_panels: List[Tuple],
        psd_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        scatter_panels: List[Tuple],
        lat: np.ndarray,
        lon: np.ndarray,
        unit: str,
        var_long_name: str,
        suptitle: str,
        out_png: str,
        cmap: str = STATE_CMAP) -> None:
    """
    Two-column layout:
      Left  (~52%): spatial maps stacked vertically (3 or 4)
      Right (~40%): PSD on top, scatter plots stacked below
    Uses GridSpec for reliable spacing.
    PNG output at 400 DPI.
    """
    n_sp = len(spatial_panels)
    n_sc = len(scatter_panels)
    if n_sp == 0:
        return

    # --- Shared color range ---------------------------------------------------
    shared_data = [d for _, d, own in spatial_panels
                   if not own and d is not None and np.any(np.isfinite(d))]
    if shared_data:
        combined = np.concatenate([np.asarray(d).ravel() for d in shared_data])
        sh_lo, sh_hi = robust_minmax(combined)
    else:
        sh_lo, sh_hi = -1.0, 1.0

    extent = [float(lon.min()), float(lon.max()),
              float(lat.min()), float(lat.max())]

    # --- Figure sizing --------------------------------------------------------
    n_right = 1 + n_sc

    fig_w = 14.0
    row_h = 3.2
    fig_h = n_sp * row_h + 1.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold",
                 y=1.0 - 0.15 / fig_h)

    # --- Outer GridSpec: 2 columns, tight margins, wide gap -------------------
    outer = mgridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.52, 0.40],
        wspace=0.30,                       # wide gap: colorbar won't hit PSD
        left=0.04, right=0.97,             # tight margins: use full page width
        top=1.0 - 0.9 / fig_h,            # room for suptitle
        bottom=0.05)

    # --- Left column: n_sp map panels stacked ---------------------------------
    gs_left = mgridspec.GridSpecFromSubplotSpec(
        n_sp, 1,
        subplot_spec=outer[0, 0],
        hspace=0.42)

    for i, (title, data2d, own_cbar) in enumerate(spatial_panels):
        ax = _add_geo_ax(fig, gs_left[i, 0], extent=extent)

        if own_cbar and data2d is not None and np.any(np.isfinite(data2d)):
            vlo, vhi = robust_minmax(data2d)
        else:
            vlo, vhi = sh_lo, sh_hi

        safe = (np.where(np.isfinite(data2d), data2d, np.nan)
                if data2d is not None
                else np.full((lat.size, lon.size), np.nan))

        kw = dict(origin="lower", extent=extent, aspect="auto",
                  vmin=vlo, vmax=vhi, cmap=cmap, interpolation="nearest")
        im = (ax.imshow(safe, transform=ccrs.PlateCarree(), **kw)
              if HAS_CARTOPY else ax.imshow(safe, **kw))

        ax.set_title(title, fontsize=FONT_PANEL_TITLE, fontweight="bold",
                     pad=8)
        cb = fig.colorbar(im, ax=ax,
                          fraction=CBAR_FRACTION, pad=CBAR_PAD,
                          shrink=CBAR_SHRINK)
        cb.ax.tick_params(labelsize=FONT_CBAR_TICK)
        cb.locator = mticker.MaxNLocator(nbins=CBAR_N_TICKS)
        cb.update_ticks()

    # --- Right column: PSD + scatter stacked ----------------------------------
    right_ratios = [0.85] + [1.0] * n_sc
    gs_right = mgridspec.GridSpecFromSubplotSpec(
        n_right, 1,
        subplot_spec=outer[0, 1],
        hspace=0.50,
        height_ratios=right_ratios)

    # PSD panel
    ax_psd = fig.add_subplot(gs_right[0, 0])
    _draw_psd(ax_psd, psd_data, var_long_name, unit)

    # Scatter panels
    for k, (title, model2d, aorc2d) in enumerate(scatter_panels):
        ax = fig.add_subplot(gs_right[1 + k, 0])
        _draw_scatter(ax, model2d, aorc2d, title)

    # --- Save PNG -------------------------------------------------------------
    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white",
                metadata={"Creator": "Earthmind evaluation"})
    plt.close(fig)
    log(f"    Saved: {out_png}")

# =============================================================================
# Data loaders
# =============================================================================

# ---- AORC
def open_aorc_year(year: int) -> xr.Dataset:
    url    = f"{AORC_S3_BASE}/{year}.zarr/"
    mapper = fsspec.get_mapper(url, anon=True)
    ds     = xr.open_zarr(mapper, consolidated=True, decode_times=False)
    start  = np.datetime64(f"{year}-01-01T00:00:00", "ns")
    end    = np.datetime64(f"{year+1}-01-01T00:00:00", "ns")
    exp    = np.arange(start, end, np.timedelta64(1, "h"))
    if ds.sizes.get("time", 0) != exp.size:
        raise RuntimeError(f"AORC time mismatch for {year}")
    return ds.assign_coords(time=("time", exp))

def aorc_hour_index(t_ns: np.datetime64, year: int) -> int:
    return int((t_ns - np.datetime64(f"{year}-01-01T00:00:00", "ns"))
                / np.timedelta64(1, "h"))

def aorc_precip_6hracc(aorc_bbox, valid_t, year,
                        tgt_lat, tgt_lon) -> np.ndarray:
    acc = np.zeros((tgt_lat.size, tgt_lon.size), dtype=np.float32)
    for dh in range(6, 0, -1):
        tt = valid_t - np.timedelta64(dh - 1, "h")
        try:
            hw  = aorc_hour_index(tt, year)
            w   = np.clip(
                aorc_bbox["APCP_surface"].isel(time=hw).interp(
                    latitude=tgt_lat, longitude=tgt_lon,
                    method="linear"
                ).values.astype(np.float32),
                0.0, None)
            acc += w
        except Exception:
            return np.full_like(acc, np.nan)
    return np.clip(acc, 0.0, None)

# ---- Earthmind
class EarthmindReader:
    def __init__(self, zarr_path: str):
        self.root = zarr.open_group(zarr_path, mode="r")
        self.lat  = np.asarray(self.root["latitude"][:],  dtype=np.float64)
        self.lon  = normalize_lon_180(
            np.asarray(self.root["longitude"][:], dtype=np.float64))
        self.time_ic    = self._robust_time_parse(self.root["time_ic"]).ravel()
        self.valid_time = self._robust_time_parse(self.root["valid_time"])
        self.lead_hours = np.asarray(self.root["lead_time"][:], dtype=np.int32)
        log(f"Earthmind: {len(self.time_ic)} ICs x {len(self.lead_hours)} leads, "
            f"grid {self.lat.size}x{self.lon.size}")

    @staticmethod
    def _robust_time_parse(zarr_arr) -> np.ndarray:
        raw   = np.asarray(zarr_arr[:])
        shape = raw.shape
        if np.issubdtype(raw.dtype, np.datetime64):
            return raw.astype("datetime64[ns]")
        if not np.issubdtype(raw.dtype, np.integer):
            return np.array([to_dt64ns(x) for x in raw.ravel()],
                            dtype="datetime64[ns]").reshape(shape)
        raw_i64   = raw.astype("int64")
        attrs     = dict(zarr_arr.attrs) if zarr_arr.attrs else {}
        units_str = attrs.get("units", "")
        if "seconds since" in units_str.lower():
            return (raw_i64 * np.int64(1_000_000_000)).astype("datetime64[ns]")
        sample  = raw_i64.ravel()
        abs_max = np.abs(sample[sample != 0]).max() if np.any(sample != 0) else 0
        if abs_max > 1e15:
            return raw_i64.astype("datetime64[ns]")
        return (raw_i64 * np.int64(1_000_000_000)).astype("datetime64[ns]")

    def get_field(self, var: str, ic_idx: int, lead_idx: int) -> xr.DataArray:
        arr    = self.root[var]
        data2d = (arr[ic_idx, lead_idx, :, :]
                  if arr.ndim == 4
                  else arr[ic_idx * len(self.lead_hours) + lead_idx, :, :])
        return xr.DataArray(
            np.asarray(data2d, dtype=np.float32),
            dims=("latitude", "longitude"),
            coords={"latitude": self.lat, "longitude": self.lon},
            name=var)

    def get_valid_time(self, ic_idx: int, lead_idx: int) -> np.datetime64:
        if self.valid_time.ndim == 2:
            return self.valid_time[ic_idx, lead_idx]
        return self.valid_time[ic_idx * len(self.lead_hours) + lead_idx]

    def precip_6hracc(self, ic_idx, lead_idx, lead_hour_to_idx,
                       tgt_lat, tgt_lon) -> np.ndarray:
        lead_h = int(self.lead_hours[lead_idx])
        acc    = np.zeros((tgt_lat.size, tgt_lon.size), dtype=np.float32)
        for dh in range(6, 0, -1):
            h = lead_h - (dh - 1)
            if h < 1 or h not in lead_hour_to_idx:
                return np.full_like(acc, np.nan)
            da  = self.get_field("APCP_surface", ic_idx, lead_hour_to_idx[h])
            acc += regrid_to_target(da, tgt_lat, tgt_lon, var="APCP_surface")
        return np.clip(acc, 0.0, None)

# ---- GraphCast
def get_gc_zarr_for_date(gc_zarr_root: str, ic_time: np.datetime64) -> str:
    root = Path(gc_zarr_root)
    if (root / ".zgroup").exists() or (root / ".zmetadata").exists():
        return str(root)
    year = int(str(np.datetime64(ic_time, "Y")))
    for top_dir in sorted(root.iterdir()):
        if not top_dir.is_dir(): continue
        parts = top_dir.name.split("_")
        try:
            y_start, y_end = int(parts[-3]), int(parts[-1])
        except (ValueError, IndexError):
            continue
        if not (y_start <= year < y_end): continue
        for sub in sorted(top_dir.iterdir()):
            if not sub.is_dir(): continue
            fd = sub / "forecasts_10d"
            if not fd.is_dir(): continue
            for cand in sorted(fd.iterdir()):
                if cand.name.endswith(".zarr") and cand.is_dir():
                    log(f"  [GC] Selected zarr for year {year}: {cand}")
                    return str(cand)
    raise FileNotFoundError(
        f"No GC zarr found for year {year} under {gc_zarr_root}.")

def open_graphcast(gc_zarr: str) -> xr.Dataset:
    ds = xr.open_zarr(gc_zarr, consolidated=False, decode_timedelta=True)
    if "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    ds = ds.assign_coords(
        longitude=("longitude", normalize_lon_180(ds["longitude"].values))
    ).sortby("longitude")
    return ds

def gc_field_at_valid_time(gc: xr.Dataset,
                            aorc_var: str,
                            valid_time: np.datetime64,
                            lead_td: np.timedelta64) -> Optional[xr.DataArray]:
    gc_var = GC_MAP.get(aorc_var)
    if gc_var is None or gc_var not in gc:
        return None
    init_time_ns = np.datetime64(str(valid_time - lead_td), "ns")
    gc_times_ns  = gc["time"].values.astype("datetime64[ns]")
    if init_time_ns not in gc_times_ns:
        return None
    lead_ns      = np.timedelta64(int(lead_td / np.timedelta64(1, "ns")), "ns")
    gc_deltas_ns = gc["prediction_timedelta"].values.astype("timedelta64[ns]")
    if lead_ns not in gc_deltas_ns:
        return None
    try:
        da = gc[gc_var].sel(
            time=init_time_ns, prediction_timedelta=lead_ns).load()
    except KeyError:
        try:
            da = gc[gc_var].sel(
                time=init_time_ns, prediction_timedelta=lead_ns,
                method="nearest",
                tolerance=np.timedelta64(60, "s")).load()
        except Exception as e:
            log(f"  [WARN] GC .sel() failed for {aorc_var}: {e}")
            return None
    if aorc_var == "APCP_surface_6hr":
        da = da * 1000.0
    da = da.clip(min=0.0)
    return da

# ---- HRRR
class HRRRReader:
    def __init__(self):
        self.fs       = fsspec.filesystem("s3", anon=True)
        self.crs_hrrr = CRS.from_proj4(
            "+proj=lcc +lat_0=38.5 +lon_0=-97.5 "
            "+lat_1=38.5 +lat_2=38.5 +datum=WGS84 +units=m +no_defs")
        self._fcst_cache: dict = {}

    def _anl_url(self, t):
        t = np.datetime64(t, "h")
        ymd = str(t)[:10].replace("-", ""); hh = str(t)[11:13]
        return f"hrrrzarr/sfc/{ymd}/{ymd}_{hh}z_anl.zarr"

    def _fcst_url(self, t):
        t = np.datetime64(t, "h")
        ymd = str(t)[:10].replace("-", ""); hh = str(t)[11:13]
        return f"hrrrzarr/sfc/{ymd}/{ymd}_{hh}z_fcst.zarr"

    def _xy_to_latlon(self, grp):
        x  = np.array(grp["projection_x_coordinate"][:], dtype=np.float64)
        y  = np.array(grp["projection_y_coordinate"][:], dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        tr = Transformer.from_crs(self.crs_hrrr, CRS.from_epsg(4326),
                                  always_xy=True)
        lon2d, lat2d = tr.transform(xx, yy)
        return lat2d, normalize_lon_180(lon2d)

    def _regrid(self, field2d: np.ndarray,
                lat2d: np.ndarray, lon2d: np.ndarray,
                tgt_lat: np.ndarray, tgt_lon: np.ndarray,
                var: str = "") -> np.ndarray:
        src_pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
        tl2d, tlt2d = np.meshgrid(tgt_lon, tgt_lat)
        tgt_pts = np.column_stack([tlt2d.ravel(), tl2d.ravel()])
        interpolated = griddata(
            src_pts, field2d.ravel(), tgt_pts, method="linear"
        )
        result = interpolated.reshape(tlt2d.shape).astype(np.float32)
        return _clip_nonneg(result, var)

    def _get_anl(self, valid_t, level, var):
        mapper = self.fs.get_mapper(self._anl_url(valid_t))
        ds     = xr.open_zarr(mapper,
                               group=f"{level}/{var}/{level}",
                               consolidated=True)
        field  = ds[var].astype("float32").values
        zg     = zarr.open_group(mapper, mode="r")
        lat2d, lon2d = self._xy_to_latlon(zg[level][var])
        return field, lat2d, lon2d

    def _load_fcst_prate(self, ic_t):
        key = str(np.datetime64(ic_t, "h"))
        if key in self._fcst_cache:
            return self._fcst_cache[key]
        log(f"  [HRRR] Loading fcst PRATE for IC={key}")
        mapper    = self.fs.get_mapper(self._fcst_url(ic_t))
        zg        = zarr.open_group(mapper, mode="r")
        prate_grp = zg["surface"]["PRATE"]
        prate_3d  = (np.array(prate_grp["surface"]["PRATE"][:],
                              dtype=np.float32) * 3600.0)
        prate_3d  = np.clip(prate_3d, 0.0, None)
        lat2d, lon2d = self._xy_to_latlon(prate_grp)
        self._fcst_cache[key] = (prate_3d, lat2d, lon2d)
        return prate_3d, lat2d, lon2d

    def get_precip_inst(self, valid_t, tgt_lat, tgt_lon) -> Optional[np.ndarray]:
        try:
            p3d, lat2d, lon2d = self._load_fcst_prate(
                valid_t - np.timedelta64(1, "h"))
            return self._regrid(p3d[0], lat2d, lon2d, tgt_lat, tgt_lon,
                                var="APCP_surface")
        except Exception as e:
            log(f"  [WARN] HRRR precip inst @ {fmt_time(valid_t)}: {e}")
            return None

    def get_precip_6hracc(self, ic_t, lead_h,
                           tgt_lat, tgt_lon) -> Optional[np.ndarray]:
        if lead_h > HRRR_MAX_FHR:
            return None
        try:
            p3d, lat2d, lon2d = self._load_fcst_prate(ic_t)
            idx_start = lead_h - 6
            idx_end   = lead_h
            if idx_start < 0 or idx_end > HRRR_MAX_FHR:
                return None
            acc = p3d[idx_start:idx_end].sum(axis=0)
            return self._regrid(acc, lat2d, lon2d, tgt_lat, tgt_lon,
                                var="APCP_surface")
        except Exception as e:
            log(f"  [WARN] HRRR 6hracc IC={fmt_time(ic_t)} lead={lead_h}: {e}")
            return None

    def get_field_on_grid(self, valid_t, aorc_var,
                           tgt_lat, tgt_lon) -> Optional[np.ndarray]:
        if aorc_var not in HRRR_MAP or aorc_var == "APCP_surface":
            return None
        level, hrrr_var = HRRR_MAP[aorc_var]
        try:
            field, lat2d, lon2d = self._get_anl(valid_t, level, hrrr_var)
        except Exception as e:
            log(f"  [WARN] HRRR {aorc_var} @ {fmt_time(valid_t)}: {e}")
            return None
        return self._regrid(field, lat2d, lon2d, tgt_lat, tgt_lon,
                            var=aorc_var)

# =============================================================================
# Regridding
# =============================================================================
def regrid_to_target(da_src: xr.DataArray,
                      tgt_lat: np.ndarray,
                      tgt_lon: np.ndarray,
                      method: str = "linear",
                      var: str = "") -> np.ndarray:
    tgt_lon_n = normalize_lon_180(tgt_lon)
    if "longitude" in da_src.coords:
        da_src = da_src.assign_coords(
            longitude=("longitude",
                        normalize_lon_180(da_src["longitude"].values)))
        result = da_src.interp(latitude=tgt_lat,
                               longitude=tgt_lon_n, method=method)
    elif "lon" in da_src.coords:
        da_src = da_src.assign_coords(
            lon=("lon", normalize_lon_180(da_src["lon"].values)))
        result = da_src.interp(lat=tgt_lat, lon=tgt_lon_n, method=method)
    else:
        raise KeyError(f"No lat/lon coords found: {list(da_src.coords)}")
    out = result.values.astype(np.float32)
    return _clip_nonneg(out, var)

# =============================================================================
# Main
# =============================================================================
def main():
    _banner = (
        f"\n{'='*64}\n"
        f"  EarthMind CONUS Evaluation Script  v{__version__}\n"
        f"  Author : {__author__}\n"
        f"  Lab    : {__lab__}\n"
        f"  Paper  : {__publication__}\n"
        f"  Run    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*64}\n"
    )
    print(_banner, flush=True)

    p = argparse.ArgumentParser(
        description=(
            "EarthMind CONUS Evaluation — compares Earthmind, HRRR, and GraphCast "
            "against AORC ground truth with spatial maps, PSD, and scatter metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--earthmind_zarr", required=True)

    gc_grp = p.add_mutually_exclusive_group(required=True)
    gc_grp.add_argument("--graphcast_zarr_root",
                        help="Root dir containing year-range GC zarr folders")
    gc_grp.add_argument("--graphcast_zarr",
                        help="Direct path to a single GC zarr")

    p.add_argument("--year",     type=int, default=2021)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--vars",     default="ALL",
                   help="Comma-separated AORC variable names, or ALL")

    p.add_argument("--lead_hours_sub6h", nargs="+", type=int,
                   default=[1, 2, 3, 4, 5],
                   help="Sub-6h leads: AORC+HRRR+EM only, instantaneous")
    p.add_argument("--lead_hours_6h", nargs="+", type=int,
                   default=[6, 12, 18, 24, 36, 48],
                   help="6h-step leads: add GC where available; APCP as 6hr acc")

    p.add_argument("--lat_min",  type=float, default=24.0)
    p.add_argument("--lat_max",  type=float, default=50.0)
    p.add_argument("--lon_min",  type=float, default=-125.0)
    p.add_argument("--lon_max",  type=float, default=-66.0)
    p.add_argument("--plot_stride",    type=int, default=3,
                   help="Spatial stride for plotting (3 = every 3rd pixel)")
    p.add_argument("--metrics_stride", type=int, default=1,
                   help="Spatial stride for metrics (1 = full resolution)")
    p.add_argument("--ic_indices", nargs="+", type=int, default=None,
                   help="Subset of IC indices to process (default: all)")
    args = p.parse_args()

    use_gc_root          = args.graphcast_zarr_root is not None
    gc_zarr_root_or_path = args.graphcast_zarr_root or args.graphcast_zarr
    vars_to_eval = (list(AORC_VARS)
                    if args.vars.strip().upper() == "ALL"
                    else [v.strip() for v in args.vars.split(",") if v.strip()])

    outdir  = ensure_dir(args.save_dir)
    plotdir = ensure_dir(outdir / "combined_plots")

    sub6h_set = set(args.lead_hours_sub6h) - {6}
    h6_set    = set(args.lead_hours_6h)
    all_leads = sorted(sub6h_set | h6_set)
    log(f"Sub-6h leads (3-panel, inst)   : {sorted(sub6h_set)}")
    log(f"6h-step leads (3/4-panel, acc) : {sorted(h6_set)}")

    log("Opening AORC...")
    aorc      = open_aorc_year(args.year)
    aorc_bbox = aorc.sel(
        latitude=slice(args.lat_min, args.lat_max),
        longitude=slice(args.lon_min, args.lon_max))
    a_lat_full = aorc_bbox["latitude"].values
    a_lon_full = aorc_bbox["longitude"].values
    ps = max(1, args.plot_stride)
    ms = max(1, args.metrics_stride)
    a_lat_p = a_lat_full[::ps]; a_lon_p = a_lon_full[::ps]
    a_lat_m = a_lat_full[::ms]; a_lon_m = a_lon_full[::ms]
    log(f"AORC grid {a_lat_full.size}x{a_lon_full.size} | "
        f"plot {a_lat_p.size}x{a_lon_p.size} | "
        f"metrics {a_lat_m.size}x{a_lon_m.size}")

    log("Opening Earthmind...")
    em = EarthmindReader(args.earthmind_zarr)

    _gc_cache: dict = {}
    def get_gc_for_ic(ic_time):
        if use_gc_root:
            try:
                zarr_path = get_gc_zarr_for_date(gc_zarr_root_or_path, ic_time)
            except FileNotFoundError as e:
                log(f"  [WARN] {e}"); return None
        else:
            zarr_path = gc_zarr_root_or_path
        if zarr_path not in _gc_cache:
            log(f"  [GC] Opening {zarr_path}")
            _gc_cache[zarr_path] = open_graphcast(zarr_path)
            tv = _gc_cache[zarr_path]["time"].values
            log(f"  [GC] {len(tv)} ICs: {tv[0]} ... {tv[-1]}")
        return _gc_cache[zarr_path]

    log("Initialising HRRR reader...")
    hrrr = HRRRReader()

    n_ic       = len(em.time_ic)
    ic_indices = (args.ic_indices if args.ic_indices is not None
                  else list(range(n_ic)))
    lead_hour_to_idx = {int(h): i for i, h in enumerate(em.lead_hours)}
    avail   = [h for h in all_leads if h in lead_hour_to_idx]
    missing = [h for h in all_leads if h not in lead_hour_to_idx]
    if missing:
        log(f"[WARN] Leads missing from EM: {missing}")
    log(f"Evaluating {len(avail)} leads: {avail}")

    log("GC variable coverage:")
    for v in AORC_VARS:
        gc_v = GC_MAP.get(v, GC_MAP.get(f"{v}_6hr"))
        note = " [removed -- MSLP != surface pressure]" if v == "PRES_surface" else ""
        log(f"  {v:30s} -> {gc_v if gc_v else 'NOT IN GC (3-panel only)' + note}")

    csv_path = outdir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_file)
    csv_w.writerow(["variable", "ic_time", "lead_hour", "valid_time",
                    "comparison", "units",
                    "N", "r", "rmse", "bias", "mae", "std_err"])

    def write_met(var, ic_t, lead_h, valid_t, comp, units, mod, tru):
        met = compute_metrics(mod, tru)
        csv_w.writerow([var, str(ic_t), lead_h, str(valid_t), comp, units,
                        met["N"],
                        f"{met['r']:.6f}",    f"{met['rmse']:.6f}",
                        f"{met['bias']:.6f}", f"{met['mae']:.6f}",
                        f"{met['std_err']:.6f}"])

    done = 0
    for ic_idx in ic_indices:
        ic_time = em.time_ic[ic_idx]
        log(f"\n{'='*60}\nIC {ic_idx+1}/{n_ic}: {fmt_time(ic_time)}\n{'='*60}")
        gc = get_gc_for_ic(ic_time)

        for lead_h in avail:
            lead_idx = lead_hour_to_idx[lead_h]
            valid_t  = em.get_valid_time(ic_idx, lead_idx)
            lead_td  = np.timedelta64(lead_h, "h")

            is_6h  = lead_h in h6_set
            has_gc = gc is not None and is_6h

            log(f"  Lead +{lead_h:3d}h  valid={fmt_time(valid_t)}  "
                f"group={'6h' if is_6h else 'sub6h'}  "
                f"GC={'yes' if has_gc else 'no'}")

            ic_str = fmt_time(ic_time).replace(":", "")
            vt_str = fmt_time(valid_t).replace(":", "")

            for var in vars_to_eval:
                if var not in em.root or var not in aorc_bbox:
                    continue
                done += 1
                is_precip  = (var == "APCP_surface")
                long_name  = VAR_LONG_NAMES.get(var, var)
                var_has_gc = var not in VARS_NO_GC and var in GC_MAP
                cmap       = PRECIP_CMAP if is_precip else STATE_CMAP
                tag        = f"{var}_IC{ic_str}_lead{lead_h:03d}h_valid{vt_str}"

                # ============================================================
                # PRECIPITATION
                # ============================================================
                if is_precip:

                    # --------------------------------------------------------
                    # INSTANTANEOUS precip
                    # --------------------------------------------------------
                    unit_inst = "mm/hr"
                    em_da     = em.get_field(var, ic_idx, lead_idx)
                    em_inst_m = regrid_to_target(em_da, a_lat_m, a_lon_m, var=var)
                    em_inst_p = regrid_to_target(em_da, a_lat_p, a_lon_p, var=var)
                    try:
                        hw = aorc_hour_index(valid_t, args.year)
                        ao_inst_m = np.clip(
                            aorc_bbox[var].isel(time=hw).interp(
                                latitude=a_lat_m, longitude=a_lon_m,
                                method="linear").values.astype(np.float32),
                            0.0, None)
                        ao_inst_p = np.clip(
                            aorc_bbox[var].isel(time=hw).interp(
                                latitude=a_lat_p, longitude=a_lon_p,
                                method="linear").values.astype(np.float32),
                            0.0, None)
                    except Exception as e:
                        log(f"    [WARN] AORC {var} @ {fmt_time(valid_t)}: {e}")
                        continue
                    hr_inst_m = hrrr.get_precip_inst(valid_t, a_lat_m, a_lon_m)
                    hr_inst_p = hrrr.get_precip_inst(valid_t, a_lat_p, a_lon_p)
                    hr_inst_m, em_inst_m = _apply_mask(
                        ao_inst_m, hr_inst_m, em_inst_m)
                    hr_inst_p, em_inst_p = _apply_mask(
                        ao_inst_p, hr_inst_p, em_inst_p)

                    em_qq_inst_m = apply_qq_correction(em_inst_m, ao_inst_m).clip(min=0.0)
                    em_qq_inst_p = apply_qq_correction(em_inst_p, ao_inst_p).clip(min=0.0)

                    psd_inst: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                    psd_inst["AORC"] = compute_radial_psd(ao_inst_p, a_lat_p, a_lon_p)
                    if hr_inst_p is not None:
                        psd_inst["HRRR"] = compute_radial_psd(hr_inst_p, a_lat_p, a_lon_p)
                    psd_inst["Earthmind"] = compute_radial_psd(em_qq_inst_p, a_lat_p, a_lon_p)

                    suptitle_inst = (
                        f"{long_name} INSTANTANEOUS ({unit_inst})  |  "
                        f"IC: {fmt_time(ic_time)}  Lead: +{lead_h}h  "
                        f"Valid: {fmt_time(valid_t)}")
                    save_combined_figure(
                        spatial_panels=[
                            ("AORC",      ao_inst_p,    False),
                            ("HRRR",      hr_inst_p,    False),
                            ("Earthmind", em_qq_inst_p, False)],
                        psd_data=psd_inst,
                        scatter_panels=[
                            ("HRRR vs AORC",  hr_inst_m,    ao_inst_m),
                            ("EM vs AORC",    em_qq_inst_m, ao_inst_m)],
                        lat=a_lat_p, lon=a_lon_p,
                        unit=unit_inst, var_long_name=long_name,
                        suptitle=suptitle_inst,
                        out_png=str(plotdir / f"{tag}_inst.png"),
                        cmap=cmap)
                    write_met(var, ic_time, lead_h, valid_t,
                              "EM_vs_AORC_inst",   unit_inst, em_qq_inst_m, ao_inst_m)
                    if hr_inst_m is not None:
                        write_met(var, ic_time, lead_h, valid_t,
                                  "HRRR_vs_AORC_inst", unit_inst, hr_inst_m, ao_inst_m)

                    # --------------------------------------------------------
                    # 6-HR ACCUMULATED precip
                    # --------------------------------------------------------
                    if is_6h:
                        unit_acc  = "mm/6h"
                        ao_acc_m = aorc_precip_6hracc(
                            aorc_bbox, valid_t, args.year, a_lat_m, a_lon_m)
                        ao_acc_p = aorc_precip_6hracc(
                            aorc_bbox, valid_t, args.year, a_lat_p, a_lon_p)
                        if not np.any(np.isfinite(ao_acc_m)):
                            log(f"    [WARN] AORC 6hracc all-NaN @ {fmt_time(valid_t)}")
                        else:
                            em_acc_m = em.precip_6hracc(
                                ic_idx, lead_idx, lead_hour_to_idx, a_lat_m, a_lon_m)
                            em_acc_p = em.precip_6hracc(
                                ic_idx, lead_idx, lead_hour_to_idx, a_lat_p, a_lon_p)
                            hr_acc_m = hrrr.get_precip_6hracc(
                                ic_time, lead_h, a_lat_m, a_lon_m)
                            hr_acc_p = hrrr.get_precip_6hracc(
                                ic_time, lead_h, a_lat_p, a_lon_p)
                            gc_acc_da = (gc_field_at_valid_time(
                                             gc, "APCP_surface_6hr", valid_t, lead_td)
                                         if has_gc else None)
                            gc_acc_m = (regrid_to_target(gc_acc_da, a_lat_m, a_lon_m,
                                                         var="APCP_surface")
                                        if gc_acc_da is not None else None)
                            gc_acc_p = (regrid_to_target(gc_acc_da, a_lat_p, a_lon_p,
                                                         var="APCP_surface")
                                        if gc_acc_da is not None else None)
                            hr_acc_m, em_acc_m, gc_acc_m = _apply_mask(
                                ao_acc_m, hr_acc_m, em_acc_m, gc_acc_m)
                            hr_acc_p, em_acc_p, gc_acc_p = _apply_mask(
                                ao_acc_p, hr_acc_p, em_acc_p, gc_acc_p)

                            em_qq_acc_m = apply_qq_correction(em_acc_m, ao_acc_m).clip(min=0.0)
                            em_qq_acc_p = apply_qq_correction(em_acc_p, ao_acc_p).clip(min=0.0)

                            psd_acc: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                            psd_acc["AORC"] = compute_radial_psd(ao_acc_p, a_lat_p, a_lon_p)
                            if hr_acc_p is not None:
                                psd_acc["HRRR"] = compute_radial_psd(hr_acc_p, a_lat_p, a_lon_p)
                            if gc_acc_p is not None:
                                psd_acc["GraphCast"] = compute_radial_psd(gc_acc_p, a_lat_p, a_lon_p)
                            psd_acc["Earthmind"] = compute_radial_psd(em_qq_acc_p, a_lat_p, a_lon_p)

                            suptitle_acc = (
                                f"{long_name} 6-HR ACCUMULATED ({unit_acc})  |  "
                                f"IC: {fmt_time(ic_time)}  Lead: +{lead_h}h  "
                                f"Valid: {fmt_time(valid_t)}")
                            if gc_acc_p is not None:
                                sp = [("AORC",      ao_acc_p,    False),
                                      ("GraphCast", gc_acc_p,    True),
                                      ("HRRR",      hr_acc_p,    False),
                                      ("Earthmind", em_qq_acc_p, False)]
                                sc = [("GC vs AORC",   gc_acc_m,    ao_acc_m),
                                      ("HRRR vs AORC", hr_acc_m,    ao_acc_m),
                                      ("EM vs AORC",   em_qq_acc_m, ao_acc_m)]
                            else:
                                sp = [("AORC",      ao_acc_p,    False),
                                      ("HRRR",      hr_acc_p,    False),
                                      ("Earthmind", em_qq_acc_p, False)]
                                sc = [("HRRR vs AORC", hr_acc_m,    ao_acc_m),
                                      ("EM vs AORC",   em_qq_acc_m, ao_acc_m)]
                            save_combined_figure(
                                spatial_panels=sp, psd_data=psd_acc,
                                scatter_panels=sc,
                                lat=a_lat_p, lon=a_lon_p,
                                unit=unit_acc, var_long_name=long_name,
                                suptitle=suptitle_acc,
                                out_png=str(plotdir / f"{tag}_6hracc.png"),
                                cmap=cmap)
                            write_met(var, ic_time, lead_h, valid_t,
                                      "EM_vs_AORC_6hracc",  unit_acc, em_qq_acc_m, ao_acc_m)
                            if hr_acc_m is not None:
                                write_met(var, ic_time, lead_h, valid_t,
                                          "HRRR_vs_AORC_6hracc", unit_acc, hr_acc_m, ao_acc_m)
                            if gc_acc_m is not None:
                                write_met(var, ic_time, lead_h, valid_t,
                                          "GC_vs_AORC_6hracc",   unit_acc, gc_acc_m, ao_acc_m)

                # ============================================================
                # STATE VARIABLES
                # ============================================================
                else:
                    unit  = VAR_UNITS.get(var, "")
                    em_da = em.get_field(var, ic_idx, lead_idx)
                    em_m  = regrid_to_target(em_da, a_lat_m, a_lon_m, var=var)
                    em_p  = regrid_to_target(em_da, a_lat_p, a_lon_p, var=var)
                    try:
                        hw   = aorc_hour_index(valid_t, args.year)
                        ao_m = _clip_nonneg(
                            aorc_bbox[var].isel(time=hw).interp(
                                latitude=a_lat_m, longitude=a_lon_m,
                                method="linear").values.astype(np.float32),
                            var)
                        ao_p = _clip_nonneg(
                            aorc_bbox[var].isel(time=hw).interp(
                                latitude=a_lat_p, longitude=a_lon_p,
                                method="linear").values.astype(np.float32),
                            var)
                    except Exception as e:
                        log(f"    [WARN] AORC {var} @ {fmt_time(valid_t)}: {e}")
                        continue
                    hr_m = hrrr.get_field_on_grid(valid_t, var, a_lat_m, a_lon_m)
                    hr_p = hrrr.get_field_on_grid(valid_t, var, a_lat_p, a_lon_p)

                    gc_m, gc_p = None, None
                    if has_gc and var_has_gc:
                        gc_da = gc_field_at_valid_time(gc, var, valid_t, lead_td)
                        if gc_da is not None:
                            gc_m = regrid_to_target(gc_da, a_lat_m, a_lon_m, var=var)
                            gc_p = regrid_to_target(gc_da, a_lat_p, a_lon_p, var=var)
                        elif var in GC_MAP:
                            log(f"    [WARN] GC returned None for {var} "
                                f"init={fmt_time(valid_t - lead_td)} "
                                f"lead=+{lead_h}h")

                    hr_m, em_m, gc_m = _apply_mask(ao_m, hr_m, em_m, gc_m)
                    hr_p, em_p, gc_p = _apply_mask(ao_p, hr_p, em_p, gc_p)
                    em_qq_m = apply_qq_correction(em_m, ao_m)
                    em_qq_p = apply_qq_correction(em_p, ao_p)

                    psd_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                    psd_data["AORC"] = compute_radial_psd(ao_p, a_lat_p, a_lon_p)
                    if hr_p is not None:
                        psd_data["HRRR"] = compute_radial_psd(hr_p, a_lat_p, a_lon_p)
                    if gc_p is not None:
                        psd_data["GraphCast"] = compute_radial_psd(gc_p, a_lat_p, a_lon_p)
                    psd_data["Earthmind"] = compute_radial_psd(em_qq_p, a_lat_p, a_lon_p)

                    suptitle = (f"{long_name} ({var}) [{unit}]  |  "
                                f"IC: {fmt_time(ic_time)}  Lead: +{lead_h}h  "
                                f"Valid: {fmt_time(valid_t)}")

                    if has_gc and var_has_gc and gc_m is not None:
                        sp = [("AORC",      ao_p,    False),
                              ("GraphCast", gc_p,    True),
                              ("HRRR",      hr_p,    False),
                              ("Earthmind", em_qq_p, False)]
                        sc = [("GC vs AORC",   gc_m,    ao_m),
                              ("HRRR vs AORC", hr_m,    ao_m),
                              ("EM vs AORC",   em_qq_m, ao_m)]
                    else:
                        sp = [("AORC",      ao_p,    False),
                              ("HRRR",      hr_p,    False),
                              ("Earthmind", em_qq_p, False)]
                        sc = [("HRRR vs AORC", hr_m,    ao_m),
                              ("EM vs AORC",   em_qq_m, ao_m)]

                    save_combined_figure(
                        spatial_panels=sp, psd_data=psd_data,
                        scatter_panels=sc,
                        lat=a_lat_p, lon=a_lon_p,
                        unit=unit, var_long_name=long_name,
                        suptitle=suptitle,
                        out_png=str(plotdir / f"{tag}.png"),
                        cmap=cmap)

                    write_met(var, ic_time, lead_h, valid_t,
                              "EM_vs_AORC",   unit, em_qq_m, ao_m)
                    if hr_m is not None:
                        write_met(var, ic_time, lead_h, valid_t,
                                  "HRRR_vs_AORC", unit, hr_m, ao_m)
                    if gc_m is not None:
                        write_met(var, ic_time, lead_h, valid_t,
                                  "GC_vs_AORC",   unit, gc_m, ao_m)

            if done % 20 == 0:
                csv_file.flush()

    csv_file.close()
    log(f"\n{'='*60}")
    log(f"DONE -- {done} variable-lead-IC combinations processed")
    log(f"  metrics : {csv_path}")
    log(f"  plots   : {plotdir}")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()