#!/usr/bin/env python3
"""
EarthMind Global Evaluation Script
===================================
Evaluates EarthMind downscaling output against StationBench ground-truth
observations (temperature only) and GraphCast (AI baseline) over any
global bounding box.

Spectral analysis uses 2D radially-averaged PSD with Hann windowing,
appropriate for regional and global (non-periodic) domains.
Reference: Harris et al. (2022, JAMES), Leinonen et al. (2020).

Ground truth  : StationBench (local zarr) — temperature (t2m) only.
                Station obs are interpolated to the model grid solely
                for PSD computation; scatter plots use raw point values.

Comparison
  • Earthmind  — all leads
  • GraphCast  — 6-h+ leads only

Layout (two-column, Nature-quality)
  Left  : spatial maps  — 3-panel (GC | EM | StationBench) or 2/1-panel
  Right : PSD (top) + scatter plots (below, temperature only)
"""
from __future__ import annotations

__author__      = "Somnath Luitel and Manmeet Singh"
__lab__         = ("AI Research Lab, Department of Earth Sciences, "
                   "Environmental and Atmospheric Sciences, "
                   "Western Kentucky University")
__publication__ = "Earthmind-Highres (2026)"
__version__     = "2.0.0"
__license__     = "<License>"

import argparse
import csv
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import zarr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
from scipy.interpolate import griddata

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn("cartopy not installed — maps will lack coastlines/borders")

# =============================================================================
# Nature-quality figure constants  (two-column layout: maps | PSD + scatter)
# =============================================================================
FIGURE_DPI = 400

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",
    "pdf.fonttype":     42,          # TrueType in PDF (Nature requirement)
    "ps.fonttype":      42,
})

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
FONT_GEO_LABEL     = 14.0

CBAR_FRACTION = 0.035
CBAR_PAD      = 0.015
CBAR_SHRINK   = 0.85
CBAR_N_TICKS  = 4

SCATTER_S     = 4.0
SCATTER_ALPHA = 0.15
SCATTER_COLOR = "steelblue"

# PSD styling — consistent with CONUS script
PSD_COLORS = {
    "StationBench": "#1b9e77",   # teal-green  (replaces AORC role)
    "GraphCast":    "#d95f02",   # orange
    "Earthmind":    "#e7298a",   # magenta-pink
}
PSD_LINEWIDTHS = {
    "StationBench": 2.2,
    "GraphCast":    1.6,
    "Earthmind":    2.0,
}
PSD_LINESTYLES = {
    "StationBench": "-",
    "GraphCast":    "--",
    "Earthmind":    "-",
}

# Station dot styling
STN_DOT_S     = 22
STN_DOT_ALPHA = 0.82
STN_DOT_EC    = "k"
STN_DOT_EW    = 0.25
STN_DOT_ZO    = 5

# =============================================================================
# Variable metadata
# =============================================================================
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

# GraphCast variable name mapping
GC_MAP = {
    "TMP_2maboveground":   "2m_temperature",
    "UGRD_10maboveground": "10m_u_component_of_wind",
    "VGRD_10maboveground": "10m_v_component_of_wind",
    "APCP_surface_6hr":    "total_precipitation_6hr",
}

# Variables with no GC counterpart
VARS_NO_GC = {
    "SPFH_2maboveground",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
}

# Variables that are non-negative by physical definition
NONNEG_VARS = {
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "SPFH_2maboveground",
}

# Only temperature has StationBench ground truth
SB_COMPARABLE_VARS = {"TMP_2maboveground"}

# StationBench variable name candidates for temperature
SB_TMP_CANDIDATES = ("t2m", "2t", "2m_temperature", "TMP_2maboveground")

PRECIP_CMAP      = "YlGnBu"
STATE_CMAP       = "RdYlBu_r"
QQ_N_QUANT       = 2000
EARTH_RADIUS_KM  = 6371.0

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
    return np.clip(arr, 0.0, None) if var in NONNEG_VARS else arr

def robust_minmax(arr, p_lo=2, p_hi=98):
    a = np.asarray(arr); a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(a, p_lo))
    hi = float(np.percentile(a, p_hi))
    if lo == hi:
        lo -= 1.0; hi += 1.0
    return lo, hi

# =============================================================================
# 2D Radially-Averaged Power Spectral Density  (Hann window)
# =============================================================================
def compute_radial_psd(
        field_2d: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radially-averaged PSD of a 2-D field on a regular lat/lon grid.
    Returns (wavelength_km, psd) arrays; both empty if insufficient data.
    """
    ny, nx = field_2d.shape
    field  = field_2d.copy().astype(np.float64)
    mask   = np.isfinite(field)
    if mask.sum() < 100:
        return np.array([]), np.array([])

    field_mean = np.nanmean(field)
    field[~mask] = field_mean
    field -= field_mean

    win_y  = np.hanning(ny)
    win_x  = np.hanning(nx)
    window = np.outer(win_y, win_x)
    window_power = np.mean(window ** 2)

    fft2  = np.fft.fft2(field * window)
    fft2s = np.fft.fftshift(fft2)
    power = (np.abs(fft2s) ** 2) / (nx * ny) / window_power

    mid_lat  = np.mean(lat)
    dlat_km  = np.abs(np.mean(np.diff(lat))) * (np.pi / 180.0) * EARTH_RADIUS_KM
    dlon_km  = (np.abs(np.mean(np.diff(lon))) * (np.pi / 180.0)
                * EARTH_RADIUS_KM * np.cos(np.deg2rad(mid_lat)))

    freq_y   = np.fft.fftshift(np.fft.fftfreq(ny, d=dlat_km))
    freq_x   = np.fft.fftshift(np.fft.fftfreq(nx, d=dlon_km))
    fx2, fy2 = np.meshgrid(freq_x, freq_y)
    freq_r   = np.sqrt(fx2 ** 2 + fy2 ** 2)

    max_freq = min(
        np.max(np.abs(freq_y[freq_y > 0])),
        np.max(np.abs(freq_x[freq_x > 0])))
    n_bins      = min(nx, ny) // 2
    freq_bins   = np.linspace(0, max_freq, n_bins + 1)
    freq_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])

    psd_radial = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        ring = (freq_r >= freq_bins[i]) & (freq_r < freq_bins[i + 1])
        if ring.sum() > 0:
            psd_radial[i] = np.mean(power[ring])

    valid       = (freq_centers > 0) & (psd_radial > 0)
    freq_valid  = freq_centers[valid]
    psd_valid   = psd_radial[valid]
    if freq_valid.size == 0:
        return np.array([]), np.array([])

    return 1.0 / freq_valid, psd_valid   # wavelength_km, psd

def stations_to_grid(
        stn_lat: np.ndarray,
        stn_lon: np.ndarray,
        stn_vals: np.ndarray,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray) -> np.ndarray:
    """
    Interpolate scattered station observations onto a regular lat/lon grid
    using linear (Delaunay) triangulation.  Used exclusively to provide a
    StationBench PSD reference curve.  Grid points far from any station
    will be NaN (handled gracefully by compute_radial_psd).
    """
    mask   = np.isfinite(stn_vals)
    pts    = np.column_stack([stn_lat[mask], stn_lon[mask]])
    vals   = stn_vals[mask].astype(np.float64)
    glon2, glat2 = np.meshgrid(grid_lon, grid_lat)
    tgt    = np.column_stack([glat2.ravel(), glon2.ravel()])
    result = griddata(pts, vals, tgt, method="linear")
    return result.reshape(glat2.shape).astype(np.float32)

# =============================================================================
# Q-Q bias correction
# =============================================================================
def apply_qq_correction(
        model_flat: np.ndarray,
        obs_at_stn: np.ndarray,
        model_at_stn: np.ndarray) -> np.ndarray:
    """
    Quantile-mapping correction: fits model@stations → obs@stations transfer
    function, then applies it to every grid point in model_flat.
    """
    mask  = np.isfinite(obs_at_stn) & np.isfinite(model_at_stn)
    obs_s = obs_at_stn[mask].astype(np.float64)
    mod_s = model_at_stn[mask].astype(np.float64)
    if obs_s.size < 10:
        log("  [QQ] Too few paired samples (<10) — correction skipped.")
        return model_flat.copy().astype(np.float32)

    q_levels = np.linspace(0.0, 100.0, QQ_N_QUANT)
    q_obs    = np.percentile(obs_s, q_levels)
    q_mod    = np.percentile(mod_s, q_levels)
    delta    = q_obs - q_mod

    mf  = np.asarray(model_flat, np.float64).ravel().copy()
    fin = np.isfinite(mf)
    mf[fin] += np.interp(mf[fin], q_mod, delta)
    log(f"  [QQ] mean Δ={float(np.mean(delta)):.3f}  "
        f"max|Δ|={float(np.max(np.abs(delta))):.3f}  N_stn={int(obs_s.size)}")
    return mf.reshape(model_flat.shape).astype(np.float32)

# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(model_vals, truth_vals) -> dict:
    x = np.asarray(model_vals).ravel().astype(np.float64)
    y = np.asarray(truth_vals).ravel().astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 10:
        return dict(N=n, r=np.nan, rmse=np.nan,
                    bias=np.nan, mae=np.nan, std_err=np.nan)
    err  = x - y
    sx, sy = float(np.std(x)), float(np.std(y))
    return dict(
        N       = n,
        r       = float(np.corrcoef(x, y)[0, 1]) if (sx > 0 and sy > 0) else np.nan,
        rmse    = float(np.sqrt(np.mean(err ** 2))),
        bias    = float(np.mean(err)),
        mae     = float(np.mean(np.abs(err))),
        std_err = float(np.std(err)),
    )

# =============================================================================
# Figure panel helpers
# =============================================================================

# ---- Geo axes
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

# ---- Colorbar
def _add_colorbar(fig, mappable, ax):
    cb = fig.colorbar(mappable, ax=ax,
                      fraction=CBAR_FRACTION, pad=CBAR_PAD,
                      shrink=CBAR_SHRINK)
    cb.ax.tick_params(labelsize=FONT_CBAR_TICK)
    cb.locator = mticker.MaxNLocator(nbins=CBAR_N_TICKS)
    # Use plain decimal formatting — no offset/power-of-ten notation
    cb.formatter = mticker.FormatStrFormatter("%.4g")
    cb.update_ticks()
    return cb

# ---- StationBench spatial panel (dots only)
def _draw_stationbench_panel(ax, fig, stn_lat, stn_lon, stn_vals,
                              title, vmin, vmax, cmap, unit,
                              extent=None):
    """
    Draws coloured station dots on a geo-axes.  No background raster.
    Extent matches model panels so all panels are geographically identical.
    """
    if extent and HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=FONT_PANEL_TITLE, fontweight="bold", pad=8)

    if stn_lat is None or stn_lat.size == 0 or not np.any(np.isfinite(stn_vals)):
        ax.text(0.5, 0.5, "No station data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return None

    mask = np.isfinite(stn_vals)
    kw   = dict(c=stn_vals[mask], s=STN_DOT_S, alpha=STN_DOT_ALPHA,
                edgecolors=STN_DOT_EC, linewidths=STN_DOT_EW,
                vmin=vmin, vmax=vmax, cmap=cmap,
                zorder=STN_DOT_ZO, rasterized=True)
    if HAS_CARTOPY:
        sc = ax.scatter(stn_lon[mask], stn_lat[mask],
                        transform=ccrs.PlateCarree(), **kw)
    else:
        sc = ax.scatter(stn_lon[mask], stn_lat[mask], **kw)

    _add_colorbar(fig, sc, ax)
    return sc

# ---- Scatter panel
def _draw_scatter(ax, model_vals, truth_vals, title,
                  xlabel="StationBench (truth)", sample_max=300_000):
    if model_vals is None or truth_vals is None:
        ax.set_title(f"{title}\n(missing)", fontsize=FONT_SCATTER_TITLE)
        ax.axis("off"); return False

    x = np.asarray(truth_vals).ravel()
    y = np.asarray(model_vals).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        ax.set_title(f"{title}\n(N<10)", fontsize=FONT_SCATTER_TITLE)
        ax.axis("off"); return False

    if x.size > sample_max:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(x.size, size=sample_max, replace=False)
        x, y = x[idx], y[idx]

    lo = min(float(np.percentile(x, 0.5)),  float(np.percentile(y, 0.5)))
    hi = max(float(np.percentile(x, 99.5)), float(np.percentile(y, 99.5)))
    if lo == hi:
        lo -= 1.0; hi += 1.0

    met = compute_metrics(y, x)
    ax.scatter(x, y, s=SCATTER_S, alpha=SCATTER_ALPHA,
               c=SCATTER_COLOR, edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel,   fontsize=FONT_SCATTER_AXIS)
    ax.set_ylabel("Model",  fontsize=FONT_SCATTER_AXIS)
    ax.set_title(title,     fontsize=FONT_SCATTER_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONT_SCATTER_TICK)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.grid(True, linewidth=0.3, alpha=0.35, color="gray", linestyle="--")
    ax.set_axisbelow(True)

    r_s = f"{met['r']:.4f}" if np.isfinite(met["r"]) else "NaN"
    ax.text(0.03, 0.97,
            f"N={met['N']:,}\nr={r_s}\nRMSE={met['rmse']:.4g}\n"
            f"Bias={met['bias']:.4g}\nMAE={met['mae']:.4g}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FONT_STATS_BOX, family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#aaaaaa", alpha=0.88, linewidth=0.5))
    return True

# ---- PSD panel
def _draw_psd(ax,
              psd_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
              var_name: str,
              unit: str) -> bool:
    drawn = False
    # Draw in a fixed order so legend is consistent across figures
    for label in ["StationBench", "GraphCast", "Earthmind"]:
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
        ax.axis("off"); return False

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
        cmap: str = STATE_CMAP,
        stn_overlay: Optional[dict] = None) -> None:
    """
    Two-column layout:
      Left  (~52%) : spatial map panels (n_sp) stacked vertically.
                     Panels with data2d=None are rendered as station-dot maps.
      Right (~40%) : PSD on top; scatter plots below (may be empty).

    Parameters
    ----------
    spatial_panels  : list of (title, data2d_or_None, own_colorbar_bool)
    psd_data        : dict  label → (wavelength_km, psd)
    scatter_panels  : list of (title, model_vals, truth_vals)
                      truth_vals are raw station values (1-D array)
    stn_overlay     : dict with keys lat, lon, vals — station observations
                      used for the StationBench dots panel
    """
    n_sp = len(spatial_panels)
    n_sc = len(scatter_panels)
    if n_sp == 0:
        return

    # Shared colour range (panels not flagged own_cbar)
    shared_data = [d for _, d, own in spatial_panels
                   if not own and d is not None and np.any(np.isfinite(d))]
    if stn_overlay is not None and stn_overlay.get("vals") is not None:
        sv = np.asarray(stn_overlay["vals"]).ravel()
        sv = sv[np.isfinite(sv)]
        if sv.size:
            shared_data.append(sv)
    combined = (np.concatenate([np.asarray(d).ravel() for d in shared_data])
                if shared_data else np.array([]))
    sh_lo, sh_hi = robust_minmax(combined) if combined.size else (-1.0, 1.0)

    extent = [float(lon.min()), float(lon.max()),
              float(lat.min()), float(lat.max())]

    # Figure sizing — left column drives the height so maps are large
    n_right  = 1 + n_sc
    row_h    = 4.2                 # taller rows → bigger spatial panels
    fig_w    = 16.0                # wider figure overall
    fig_h    = max(n_sp * row_h + 1.6, n_right * row_h + 1.6)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold",
                 y=1.0 - 0.12 / fig_h)

    # Outer GridSpec: left gets more space for maps
    outer = mgridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.62, 0.34],   # wider left, narrower right
        wspace=0.38,
        left=0.04, right=0.97,
        top=1.0 - 0.75 / fig_h,
        bottom=0.05)

    # Left column: spatial panels
    gs_left = mgridspec.GridSpecFromSubplotSpec(
        n_sp, 1, subplot_spec=outer[0, 0], hspace=0.42)

    stn_lat  = (stn_overlay or {}).get("lat",  None)
    stn_lon  = (stn_overlay or {}).get("lon",  None)
    stn_vals = (stn_overlay or {}).get("vals", None)

    for i, (title, data2d, own_cbar) in enumerate(spatial_panels):
        ax = _add_geo_ax(fig, gs_left[i, 0], extent=extent)

        if data2d is None:
            # StationBench dots panel — same extent as model panels
            _draw_stationbench_panel(
                ax, fig, stn_lat, stn_lon, stn_vals,
                title, sh_lo, sh_hi, cmap, unit, extent=extent)
            continue

        vlo, vhi = (robust_minmax(data2d)
                    if own_cbar and np.any(np.isfinite(data2d))
                    else (sh_lo, sh_hi))
        safe = np.where(np.isfinite(data2d), data2d, np.nan)
        kw   = dict(origin="lower", extent=extent,
                    vmin=vlo, vmax=vhi, cmap=cmap, interpolation="nearest")
        if HAS_CARTOPY:
            im = ax.imshow(safe, transform=ccrs.PlateCarree(), **kw)
        else:
            im = ax.imshow(safe, aspect="auto", **kw)

        ax.set_title(title, fontsize=FONT_PANEL_TITLE, fontweight="bold", pad=8)
        _add_colorbar(fig, im, ax)

    # Right column: PSD + scatter
    # When no scatter panels, constrain PSD to middle 70% of right column
    # to prevent the panel stretching vertically across the full figure height.
    if n_sc == 0:
        gs_right = mgridspec.GridSpecFromSubplotSpec(
            3, 1,
            subplot_spec=outer[0, 1],
            hspace=0.30,
            height_ratios=[0.15, 0.70, 0.15])
        ax_psd = fig.add_subplot(gs_right[1, 0])
    else:
        right_ratios = [0.85] + [1.0] * n_sc
        gs_right = mgridspec.GridSpecFromSubplotSpec(
            n_right, 1,
            subplot_spec=outer[0, 1],
            hspace=0.50,
            height_ratios=right_ratios)
        ax_psd = fig.add_subplot(gs_right[0, 0])

    _draw_psd(ax_psd, psd_data, var_long_name, unit)

    for k, (title, model_vals, truth_vals) in enumerate(scatter_panels):
        ax_sc = fig.add_subplot(gs_right[1 + k, 0])
        _draw_scatter(ax_sc, model_vals, truth_vals, title)

    fig.savefig(out_png, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white",
                metadata={"Creator": "EarthMind global evaluation"})
    plt.close(fig)
    log(f"    Saved: {out_png}")

# =============================================================================
# StationBench reader
# =============================================================================
class StationBenchReader:
    """
    Opens a local StationBench zarr.
    Detects the temperature variable name automatically from SB_TMP_CANDIDATES.
    """
    def __init__(self, zarr_path: str):
        log(f"Opening StationBench: {zarr_path}")
        self.ds    = xr.open_zarr(zarr_path, consolidated=False)
        self.lat   = np.asarray(self.ds["latitude"].values,  dtype=np.float64)
        self.lon   = normalize_lon_180(
            np.asarray(self.ds["longitude"].values, dtype=np.float64))
        self.times = self.ds["time"].values.astype("datetime64[ns]")

        # Auto-detect temperature variable
        self.tmp_var = None
        for cand in SB_TMP_CANDIDATES:
            if cand in self.ds:
                self.tmp_var = cand
                log(f"  StationBench temperature variable: '{cand}'")
                break
        if self.tmp_var is None:
            warnings.warn(
                f"StationBench: none of {SB_TMP_CANDIDATES} found. "
                f"Available vars: {list(self.ds.data_vars)}")

        log(f"  StationBench: {self.lat.size} stations | "
            f"{self.times.size} time steps "
            f"({fmt_time(self.times[0])} … {fmt_time(self.times[-1])})")

    def obs_at_time(
            self,
            valid_t: np.datetime64,
            lat_min: float, lat_max: float,
            lon_min: float, lon_max: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (stn_lat, stn_lon, stn_t2m) inside bbox for the nearest
        available time step.  Returns (None, None, None) on failure.
        """
        if self.tmp_var is None:
            return None, None, None

        t_ns    = np.datetime64(valid_t, "ns")
        idx_arr = np.where(self.times == t_ns)[0]
        if idx_arr.size == 0:
            delta   = np.abs(self.times.astype("int64")
                             - np.int64(t_ns.astype("int64")))
            nearest = int(delta.argmin())
            tol_ns  = int(np.timedelta64(30, "m") / np.timedelta64(1, "ns"))
            if int(delta[nearest]) > tol_ns:
                log(f"  [SB] No obs within ±30 min of {fmt_time(valid_t)}")
                return None, None, None
            t_idx = nearest
        else:
            t_idx = int(idx_arr[0])

        tmp  = self.ds[self.tmp_var].isel(time=t_idx).values.astype(np.float64)
        bbox = ((self.lat >= lat_min) & (self.lat <= lat_max) &
                (self.lon >= lon_min) & (self.lon <= lon_max) &
                np.isfinite(tmp))
        n = int(bbox.sum())
        if n == 0:
            log(f"  [SB] 0 stations in bbox at {fmt_time(valid_t)}")
            return None, None, None
        log(f"  [SB] {n} stations in bbox at {fmt_time(valid_t)}")
        return (self.lat[bbox].astype(np.float32),
                self.lon[bbox].astype(np.float32),
                tmp[bbox].astype(np.float32))

# =============================================================================
# Earthmind reader
# =============================================================================
class EarthmindReader:
    _FILL_ABS_THRESH = 1e10

    def __init__(self, zarr_path: str):
        self.root       = zarr.open_group(zarr_path, mode="r")
        self.lat        = np.asarray(self.root["latitude"][:],  dtype=np.float64)
        self.lon        = normalize_lon_180(
            np.asarray(self.root["longitude"][:], dtype=np.float64))
        self.time_ic    = self._parse_times(self.root["time_ic"]).ravel()
        self.valid_time = self._parse_times(self.root["valid_time"])
        self.lead_hours = np.asarray(self.root["lead_time"][:], dtype=np.int32)
        log(f"Earthmind: {len(self.time_ic)} ICs × {len(self.lead_hours)} leads, "
            f"grid {self.lat.size}×{self.lon.size}")

    @staticmethod
    def _parse_times(zarr_arr) -> np.ndarray:
        raw   = np.asarray(zarr_arr[:])
        shape = raw.shape
        if np.issubdtype(raw.dtype, np.datetime64):
            return raw.astype("datetime64[ns]")
        if not np.issubdtype(raw.dtype, np.integer):
            return np.array([to_dt64ns(x) for x in raw.ravel()],
                            dtype="datetime64[ns]").reshape(shape)
        attrs = dict(zarr_arr.attrs) if zarr_arr.attrs else {}
        if "seconds since" in attrs.get("units", "").lower():
            return (raw.astype("int64") * np.int64(1_000_000_000)
                    ).astype("datetime64[ns]")
        return (raw.astype("int64") * np.int64(1_000_000_000)
                ).astype("datetime64[ns]")

    def get_field(self, var: str, ic_idx: int, lead_idx: int) -> xr.DataArray:
        arr    = self.root[var]
        data2d = (arr[ic_idx, lead_idx, :, :]
                  if arr.ndim == 4
                  else arr[ic_idx * len(self.lead_hours) + lead_idx, :, :])
        raw    = np.asarray(data2d, np.float32)

        attrs  = dict(arr.attrs) if arr.attrs else {}
        fv     = attrs.get("_FillValue", attrs.get("missing_value", None))
        if fv is not None:
            try:
                raw = np.where(np.isclose(raw, float(fv), rtol=1e-4),
                               np.nan, raw)
            except (TypeError, ValueError):
                pass
        raw = np.where(np.abs(raw) > self._FILL_ABS_THRESH, np.nan, raw)

        n_fill = int(np.sum(~np.isfinite(raw)))
        if n_fill:
            log(f"    [FILL] {var}: masked {n_fill} fill/missing values")

        return xr.DataArray(
            raw,
            dims=("latitude", "longitude"),
            coords={"latitude": self.lat, "longitude": self.lon},
            name=var)

    def get_valid_time(self, ic_idx: int, lead_idx: int) -> np.datetime64:
        return (self.valid_time[ic_idx, lead_idx]
                if self.valid_time.ndim == 2
                else self.valid_time[ic_idx * len(self.lead_hours) + lead_idx])

    def precip_6hracc(self, ic_idx, lead_idx, lead_hour_to_idx,
                       tgt_lat, tgt_lon) -> np.ndarray:
        lead_h = int(self.lead_hours[lead_idx])
        acc    = np.zeros((tgt_lat.size, tgt_lon.size), np.float32)
        for dh in range(6, 0, -1):
            h = lead_h - (dh - 1)
            if h < 1 or h not in lead_hour_to_idx:
                return np.full_like(acc, np.nan)
            da  = self.get_field("APCP_surface", ic_idx, lead_hour_to_idx[h])
            acc += (da.sel(latitude=tgt_lat, longitude=tgt_lon, method="nearest")
                      .values.astype(np.float32))
        return np.clip(acc, 0.0, None)

# =============================================================================
# GraphCast helpers
# =============================================================================
def get_gc_zarr_for_date(gc_root: str, ic_time: np.datetime64) -> str:
    root = Path(gc_root)
    if (root / ".zgroup").exists() or (root / ".zmetadata").exists():
        return str(root)
    year = int(str(np.datetime64(ic_time, "Y")))
    for top in sorted(root.iterdir()):
        if not top.is_dir(): continue
        parts = top.name.split("_")
        try:
            y_s, y_e = int(parts[-3]), int(parts[-1])
        except (ValueError, IndexError):
            continue
        if not (y_s <= year < y_e): continue
        for sub in sorted(top.iterdir()):
            fd = sub / "forecasts_10d"
            if not fd.is_dir(): continue
            for cand in sorted(fd.iterdir()):
                if cand.name.endswith(".zarr") and cand.is_dir():
                    log(f"  [GC] zarr for year {year}: {cand}")
                    return str(cand)
    raise FileNotFoundError(
        f"No GC zarr found for year {year} under {gc_root}.")

def open_graphcast(gc_zarr: str) -> xr.Dataset:
    ds = xr.open_zarr(gc_zarr, consolidated=False, decode_timedelta=True)
    if "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    return (ds.assign_coords(
                longitude=("longitude",
                           normalize_lon_180(ds["longitude"].values)))
              .sortby("longitude"))

def gc_field_at_valid_time(
        gc: xr.Dataset,
        aorc_var: str,
        valid_time: np.datetime64,
        lead_td: np.timedelta64) -> Optional[xr.DataArray]:
    gc_var = GC_MAP.get(aorc_var)
    if gc_var is None or gc_var not in gc:
        return None
    init_ns = np.datetime64(str(valid_time - lead_td), "ns")
    if init_ns not in gc["time"].values.astype("datetime64[ns]"):
        return None
    lead_ns = np.timedelta64(int(lead_td / np.timedelta64(1, "ns")), "ns")
    if lead_ns not in gc["prediction_timedelta"].values.astype("timedelta64[ns]"):
        return None
    try:
        da = gc[gc_var].sel(time=init_ns, prediction_timedelta=lead_ns).load()
    except KeyError:
        try:
            da = gc[gc_var].sel(
                time=init_ns, prediction_timedelta=lead_ns,
                method="nearest",
                tolerance=np.timedelta64(60, "s")).load()
        except Exception as e:
            log(f"  [WARN] GC .sel() failed for {aorc_var}: {e}")
            return None
    if aorc_var == "APCP_surface_6hr":
        da = da * 1000.0
    return da.clip(min=0.0)

# =============================================================================
# Regridding helpers
# =============================================================================
def regrid_to_target(
        da_src: xr.DataArray,
        tgt_lat: np.ndarray,
        tgt_lon: np.ndarray,
        method: str = "linear",
        var: str = "") -> np.ndarray:
    """Bilinearly interpolate da_src onto tgt_lat / tgt_lon."""
    tgt_lon_n = normalize_lon_180(tgt_lon)
    lat_c = "latitude" if "latitude" in da_src.coords else "lat"
    lon_c = "longitude" if "longitude" in da_src.coords else "lon"
    da_src = da_src.assign_coords(
        {lon_c: normalize_lon_180(da_src[lon_c].values.astype(np.float64))})
    out = (da_src.interp({lat_c: tgt_lat, lon_c: tgt_lon_n}, method=method)
                 .values.astype(np.float32))
    return _clip_nonneg(out, var)

def em_slice_plot(da: xr.DataArray,
                  tgt_lat: np.ndarray,
                  tgt_lon: np.ndarray) -> np.ndarray:
    """
    Extract Earthmind values by direct nearest-neighbour selection on the
    native grid.  tgt_lat/tgt_lon are a strided subset of em.lat/em.lon
    so no interpolation artefacts are introduced.
    """
    lat_c = "latitude" if "latitude" in da.coords else "lat"
    lon_c = "longitude" if "longitude" in da.coords else "lon"
    return (da.sel({lat_c: tgt_lat, lon_c: tgt_lon}, method="nearest")
              .values.astype(np.float32))

def model_at_stations(da: xr.DataArray,
                       stn_lat: np.ndarray,
                       stn_lon: np.ndarray) -> np.ndarray:
    """Bilinearly sample a model DataArray at raw station locations."""
    lat_c   = "latitude" if "latitude" in da.coords else "lat"
    lon_c   = "longitude" if "longitude" in da.coords else "lon"
    lat_pts = xr.DataArray(stn_lat.astype(np.float64), dims="station")
    lon_pts = xr.DataArray(normalize_lon_180(stn_lon.astype(np.float64)),
                           dims="station")
    da_n    = da.assign_coords(
        {lon_c: normalize_lon_180(da[lon_c].values.astype(np.float64))})
    return (da_n.interp({lat_c: lat_pts, lon_c: lon_pts}, method="linear")
                .values.astype(np.float32))

# =============================================================================
# Main
# =============================================================================
def main():
    _banner = (
        f"\n{'='*64}\n"
        f"  EarthMind Global Evaluation Script  v{__version__}\n"
        f"  Author : {__author__}\n"
        f"  Lab    : {__lab__}\n"
        f"  Paper  : {__publication__}\n"
        f"  Run    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*64}\n"
    )
    print(_banner, flush=True)

    p = argparse.ArgumentParser(
        description=(
            "EarthMind Global Evaluation — compares Earthmind and GraphCast "
            "against StationBench ground-truth observations with spatial maps, "
            "PSD curves, and scatter metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--earthmind_zarr",    required=True,
                   help="Path to Earthmind output zarr")
    p.add_argument("--stationbench_zarr", required=True,
                   help="Path to StationBench local zarr")
    p.add_argument("--save_dir",          required=True,
                   help="Output directory for plots and metrics CSV")

    gc_grp = p.add_mutually_exclusive_group(required=True)
    gc_grp.add_argument("--graphcast_zarr_root",
                        help="Root dir containing year-range GC zarr folders")
    gc_grp.add_argument("--graphcast_zarr",
                        help="Direct path to a single GC zarr")

    p.add_argument("--vars",             default="ALL",
                   help="Comma-separated variable names, or ALL")
    p.add_argument("--lead_hours_sub6h", nargs="+", type=int,
                   default=[1, 2, 3, 4, 5],
                   help="Sub-6h leads: Earthmind only, no GC")
    p.add_argument("--lead_hours_6h",    nargs="+", type=int,
                   default=[6, 12, 18, 24, 30, 36, 48],
                   help="6h-step leads: GC available; APCP as 6hr acc")
    p.add_argument("--lat_min",     type=float, default=6.0)
    p.add_argument("--lat_max",     type=float, default=36.0)
    p.add_argument("--lon_min",     type=float, default=67.0)
    p.add_argument("--lon_max",     type=float, default=97.0)
    p.add_argument("--plot_stride", type=int,   default=3,
                   help="Spatial stride for plotting (3 = every 3rd pixel)")
    p.add_argument("--ic_indices",  nargs="+",  type=int, default=None,
                   help="Subset of IC indices (default: all)")
    args = p.parse_args()

    use_gc_root     = args.graphcast_zarr_root is not None
    gc_root_or_path = args.graphcast_zarr_root or args.graphcast_zarr
    vars_to_eval    = (list(AORC_VARS)
                       if args.vars.strip().upper() == "ALL"
                       else [v.strip() for v in args.vars.split(",")
                             if v.strip()])

    outdir  = ensure_dir(args.save_dir)
    plotdir = ensure_dir(outdir / "combined_plots")

    sub6h_set = set(args.lead_hours_sub6h) - {6}
    h6_set    = set(args.lead_hours_6h)
    all_leads = sorted(sub6h_set | h6_set)
    log(f"Sub-6h leads (EM only)         : {sorted(sub6h_set)}")
    log(f"6h-step leads (GC available)   : {sorted(h6_set)}")
    log(f"Bbox : lat [{args.lat_min}, {args.lat_max}]  "
        f"lon [{args.lon_min}, {args.lon_max}]")

    # ── Open datasets ─────────────────────────────────────────────────────────
    log("Opening Earthmind...")
    em = EarthmindReader(args.earthmind_zarr)

    ps      = max(1, args.plot_stride)
    a_lat_p = em.lat[(em.lat >= args.lat_min) & (em.lat <= args.lat_max)][::ps]
    a_lon_p = em.lon[(em.lon >= args.lon_min) & (em.lon <= args.lon_max)][::ps]
    if a_lat_p.size == 0 or a_lon_p.size == 0:
        raise ValueError("No Earthmind grid points inside bbox — "
                         "check --lat_min/max and --lon_min/max.")
    log(f"EM plot grid : {a_lat_p.size}×{a_lon_p.size} (stride={ps})")

    log("Opening StationBench...")
    sb = StationBenchReader(args.stationbench_zarr)

    _gc_cache: dict = {}
    def get_gc(ic_time):
        try:
            zarr_path = (get_gc_zarr_for_date(gc_root_or_path, ic_time)
                         if use_gc_root else gc_root_or_path)
        except FileNotFoundError as e:
            log(f"  [WARN] {e}"); return None
        if zarr_path not in _gc_cache:
            log(f"  [GC] Opening {zarr_path}")
            _gc_cache[zarr_path] = open_graphcast(zarr_path)
            tv = _gc_cache[zarr_path]["time"].values
            log(f"  [GC] {len(tv)} ICs: {tv[0]} … {tv[-1]}")
        return _gc_cache[zarr_path]

    n_ic             = len(em.time_ic)
    ic_indices       = (args.ic_indices if args.ic_indices is not None
                        else list(range(n_ic)))
    lead_hour_to_idx = {int(h): i for i, h in enumerate(em.lead_hours)}
    avail            = [h for h in all_leads if h in lead_hour_to_idx]
    missing          = [h for h in all_leads if h not in lead_hour_to_idx]
    if missing:
        log(f"[WARN] Leads missing from EM: {missing}")
    log(f"Evaluating {len(avail)} leads: {avail}")

    # ── CSV metrics ───────────────────────────────────────────────────────────
    csv_path = outdir / "metrics_global.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_file)
    csv_w.writerow(["variable", "ic_time", "lead_hour", "valid_time",
                    "comparison", "units", "n_stations",
                    "N", "r", "rmse", "bias", "mae", "std_err"])

    def write_met(var, ic_t, lead_h, valid_t, comp, units,
                  mv, tv, n_stations=0):
        met = compute_metrics(mv, tv)
        csv_w.writerow([
            var, str(ic_t), lead_h, str(valid_t), comp, units, n_stations,
            met["N"],
            f"{met['r']:.6f}",    f"{met['rmse']:.6f}",
            f"{met['bias']:.6f}", f"{met['mae']:.6f}",
            f"{met['std_err']:.6f}",
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # Main loop
    # ═════════════════════════════════════════════════════════════════════════
    done = 0
    for ic_idx in ic_indices:
        ic_time = em.time_ic[ic_idx]
        log(f"\n{'='*60}\nIC {ic_idx+1}/{n_ic}: {fmt_time(ic_time)}\n{'='*60}")
        gc = get_gc(ic_time)

        for lead_h in avail:
            lead_idx = lead_hour_to_idx[lead_h]
            valid_t  = em.get_valid_time(ic_idx, lead_idx)
            lead_td  = np.timedelta64(lead_h, "h")
            is_6h    = lead_h in h6_set
            has_gc   = gc is not None and is_6h

            log(f"  Lead +{lead_h:3d}h  valid={fmt_time(valid_t)}  "
                f"GC={'yes' if has_gc else 'no'}")

            ic_str = fmt_time(ic_time).replace(":", "")
            vt_str = fmt_time(valid_t).replace(":", "")

            for var in vars_to_eval:
                if var not in em.root:
                    log(f"    [SKIP] {var} not in EM zarr"); continue
                done    += 1
                is_tmp   = (var == "TMP_2maboveground")
                is_prcp  = (var == "APCP_surface")
                unit     = VAR_UNITS.get(var, "")
                lname    = VAR_LONG_NAMES.get(var, var)
                cmap     = PRECIP_CMAP if is_prcp else STATE_CMAP
                tag      = f"{var}_IC{ic_str}_lead{lead_h:03d}h_valid{vt_str}"
                has_sb   = is_tmp    # StationBench only for temperature
                var_in_gc = (var not in VARS_NO_GC) and (var in GC_MAP)

                # ── Get Earthmind field ───────────────────────────────────────
                em_da = em.get_field(var, ic_idx, lead_idx)
                em_p  = em_slice_plot(em_da, a_lat_p, a_lon_p)

                # ── Get StationBench obs (temperature only) ───────────────────
                stn_lat = stn_lon = stn_tmp = None
                if has_sb:
                    stn_lat, stn_lon, stn_tmp = sb.obs_at_time(
                        valid_t,
                        args.lat_min, args.lat_max,
                        args.lon_min, args.lon_max)

                # ── QQ-correct Earthmind vs StationBench (temperature only) ───
                em_at_stn = None
                if has_sb and stn_lat is not None:
                    em_at_stn = model_at_stations(em_da, stn_lat, stn_lon)
                    em_p_corr = apply_qq_correction(
                        em_p.ravel(), stn_tmp, em_at_stn
                    ).reshape(em_p.shape)
                    em_at_stn_corr = apply_qq_correction(
                        em_at_stn, stn_tmp, em_at_stn)
                else:
                    em_p_corr      = em_p
                    em_at_stn_corr = em_at_stn

                # ── GraphCast field ───────────────────────────────────────────
                gc_p = gc_at_stn = None
                if has_gc and var_in_gc:
                    gc_da = gc_field_at_valid_time(gc, var, valid_t, lead_td)
                    if gc_da is not None:
                        gc_p = regrid_to_target(
                            gc_da, a_lat_p, a_lon_p, var=var)
                        if has_sb and stn_lat is not None:
                            gc_at_stn = model_at_stations(
                                gc_da, stn_lat, stn_lon)
                    else:
                        log(f"    [WARN] GC returned None for {var} "
                            f"init={fmt_time(valid_t - lead_td)} "
                            f"lead=+{lead_h}h")

                # ─────────────────────────────────────────────────────────────
                # TEMPERATURE — full treatment with StationBench
                # ─────────────────────────────────────────────────────────────
                if is_tmp:
                    # PSD: interpolate station obs to grid for SB PSD curve
                    psd_data: Dict = {}
                    if stn_lat is not None and stn_tmp is not None:
                        sb_gridded = stations_to_grid(
                            stn_lat, stn_lon, stn_tmp, a_lat_p, a_lon_p)
                        psd_data["StationBench"] = compute_radial_psd(
                            sb_gridded, a_lat_p, a_lon_p)
                    if gc_p is not None:
                        psd_data["GraphCast"] = compute_radial_psd(
                            gc_p, a_lat_p, a_lon_p)
                    psd_data["Earthmind"] = compute_radial_psd(
                        em_p_corr, a_lat_p, a_lon_p)

                    # Spatial panels
                    stn_overlay = dict(lat=stn_lat, lon=stn_lon, vals=stn_tmp)
                    if has_gc and gc_p is not None:
                        sp = [
                            ("GraphCast",              gc_p,      True),
                            ("Earthmind",              em_p_corr, False),
                            ("StationBench (raw obs)", None,      True),
                        ]
                        sc = [
                            ("GC vs StationBench", gc_at_stn,     stn_tmp),
                            ("EM vs StationBench", em_at_stn_corr, stn_tmp),
                        ]
                    else:
                        sp = [
                            ("Earthmind",              em_p_corr, False),
                            ("StationBench (raw obs)", None,      True),
                        ]
                        sc = [("EM vs StationBench", em_at_stn_corr, stn_tmp)]

                    suptitle = (
                        f"{lname} ({var}) [{unit}]  |  "
                        f"IC: {fmt_time(ic_time)}  Lead: +{lead_h}h  "
                        f"Valid: {fmt_time(valid_t)}")
                    save_combined_figure(
                        spatial_panels=sp,
                        psd_data=psd_data,
                        scatter_panels=sc,
                        lat=a_lat_p, lon=a_lon_p,
                        unit=unit, var_long_name=lname,
                        suptitle=suptitle,
                        out_png=str(plotdir / f"{tag}.png"),
                        cmap=cmap,
                        stn_overlay=stn_overlay)

                    n_s = int(stn_lat.size) if stn_lat is not None else 0
                    if em_at_stn_corr is not None and stn_tmp is not None:
                        write_met(var, ic_time, lead_h, valid_t,
                                  "EM_vs_SB_QQ", unit,
                                  em_at_stn_corr, stn_tmp, n_s)
                    if gc_at_stn is not None and stn_tmp is not None:
                        write_met(var, ic_time, lead_h, valid_t,
                                  "GC_vs_SB", unit,
                                  gc_at_stn, stn_tmp, n_s)

                # ─────────────────────────────────────────────────────────────
                # PRECIPITATION — instantaneous + 6-hr accumulated
                # ─────────────────────────────────────────────────────────────
                elif is_prcp:
                    # Instantaneous
                    psd_inst: Dict = {}
                    psd_inst["Earthmind"] = compute_radial_psd(
                        np.clip(em_p, 0.0, None), a_lat_p, a_lon_p)

                    suptitle_inst = (
                        f"{lname} INSTANTANEOUS (mm/hr)  |  "
                        f"IC: {fmt_time(ic_time)}  Lead: +{lead_h}h  "
                        f"Valid: {fmt_time(valid_t)}")
                    save_combined_figure(
                        spatial_panels=[
                            ("Earthmind", np.clip(em_p, 0.0, None), False)],
                        psd_data=psd_inst,
                        scatter_panels=[],
                        lat=a_lat_p, lon=a_lon_p,
                        unit="mm/hr", var_long_name=lname,
                        suptitle=suptitle_inst,
                        out_png=str(plotdir / f"{tag}_inst.png"),
                        cmap=cmap)

                    # 6-hr accumulated
                    if is_6h:
                        em_acc = np.clip(
                            em.precip_6hracc(
                                ic_idx, lead_idx, lead_hour_to_idx,
                                a_lat_p, a_lon_p),
                            0.0, None)
                        gc_acc_p = None
                        if has_gc:
                            gc_acc_da = gc_field_at_valid_time(
                                gc, "APCP_surface_6hr", valid_t, lead_td)
                            if gc_acc_da is not None:
                                gc_acc_p = np.clip(
                                    regrid_to_target(
                                        gc_acc_da, a_lat_p, a_lon_p,
                                        var="APCP_surface"),
                                    0.0, None)

                        psd_acc: Dict = {}
                        if gc_acc_p is not None:
                            psd_acc["GraphCast"] = compute_radial_psd(
                                gc_acc_p, a_lat_p, a_lon_p)
                        psd_acc["Earthmind"] = compute_radial_psd(
                            em_acc, a_lat_p, a_lon_p)

                        sp = ([("GraphCast", gc_acc_p, True),
                               ("Earthmind", em_acc,   False)]
                              if gc_acc_p is not None
                              else [("Earthmind", em_acc, False)])
                        suptitle_acc = (
                            f"{lname} 6-HR ACCUMULATED (mm/6h)  |  "
                            f"IC: {fmt_time(ic_time)}  Lead: +{lead_h}h  "
                            f"Valid: {fmt_time(valid_t)}")
                        save_combined_figure(
                            spatial_panels=sp,
                            psd_data=psd_acc,
                            scatter_panels=[],
                            lat=a_lat_p, lon=a_lon_p,
                            unit="mm/6h", var_long_name=lname,
                            suptitle=suptitle_acc,
                            out_png=str(plotdir / f"{tag}_6hracc.png"),
                            cmap=cmap)

                # ─────────────────────────────────────────────────────────────
                # ALL OTHER STATE VARIABLES
                # ─────────────────────────────────────────────────────────────
                else:
                    psd_data_: Dict = {}
                    if gc_p is not None:
                        psd_data_["GraphCast"] = compute_radial_psd(
                            gc_p, a_lat_p, a_lon_p)
                    psd_data_["Earthmind"] = compute_radial_psd(
                        em_p, a_lat_p, a_lon_p)

                    sp = ([("GraphCast", gc_p, True),
                           ("Earthmind", em_p, False)]
                          if gc_p is not None
                          else [("Earthmind", em_p, False)])

                    suptitle = (
                        f"{lname} ({var}) [{unit}]  |  "
                        f"IC: {fmt_time(ic_time)}  Lead: +{lead_h}h  "
                        f"Valid: {fmt_time(valid_t)}")
                    save_combined_figure(
                        spatial_panels=sp,
                        psd_data=psd_data_,
                        scatter_panels=[],
                        lat=a_lat_p, lon=a_lon_p,
                        unit=unit, var_long_name=lname,
                        suptitle=suptitle,
                        out_png=str(plotdir / f"{tag}.png"),
                        cmap=cmap)

                if done % 20 == 0:
                    csv_file.flush()

    csv_file.close()
    log(f"\n{'='*60}")
    log(f"DONE — {done} variable-lead-IC combinations processed")
    log(f"  metrics : {csv_path}")
    log(f"  plots   : {plotdir}")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()