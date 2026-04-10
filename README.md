# EarthMind-SR

**EarthMind-SR** is a generative atmospheric super-resolution framework that translates coarse-resolution (0.25°) GraphCast global weather model outputs into kilometer-scale (1 km) gridded analyses. Built on a 3-D UNet conditioned with a Latent Consistency Model (LCM) scheduler, EarthMind-SR produces high-fidelity multi-variable predictions at substantially reduced inference cost compared to conventional diffusion models.

---

## Paper

**EarthMind-SR: Kilometer-Scale Atmospheric Super-Resolution via Latent Consistency Diffusion**

Somnath Luitel\* and Manmeet Singh\* (\*Equal contribution)

AI Research Lab, Department of Earth, Environmental, and Atmospheric Sciences,
Western Kentucky University, Bowling Green, KY, USA

*Preprint — arXiv:2026.XXXXX*

### Abstract

This study presents EarthMind-SR, a generative atmospheric super-resolution framework that translates coarse-resolution (0.25°) outputs from the GraphCast global weather model into kilometer-scale (1 km) gridded analyses over the contiguous United States (CONUS). EarthMind-SR is built upon a three-dimensional UNet conditioned with a Latent Consistency Model (LCM) scheduler, enabling high-fidelity multi-variable predictions at substantially reduced inference cost compared to conventional diffusion models. The model ingests 17 atmospheric conditioning variables from GraphCast — spanning surface fields, pressure-level dynamics at 850, 700, and 500 hPa, and geopotential — augmented with static terrain descriptors (normalized elevation and sky-view factor) and instantaneous solar geometry (cosine of solar zenith angle), yielding 20 conditioning channels in total. EarthMind-SR produces simultaneous 67-hour predictions of eight AORC (Analysis of Record for Calibration) variables: precipitation, downward shortwave and longwave radiation, surface pressure, specific humidity, 2-m temperature, and 10-m wind components. During training, the model learns to map coarse GraphCast fields to AORC ground truth at 1 km resolution using a patch-based strategy (64×64 pixels) over the 2021 calendar year. At inference, patch size is expanded to 256×256 with a stride of 192, and overlapping predictions are merged via cosine-tapered spatial blending to ensure seamless, artifact-free output across the full CONUS domain. Evaluation against independent HRRR (High-Resolution Rapid Refresh) forecasts and AORC observations demonstrates that EarthMind-SR captures fine-scale spatial detail in temperature, humidity, precipitation, and radiation fields that is unresolvable at the 0.25° input scale. These results establish EarthMind-SR as a practical, computationally efficient foundation for kilometer-scale statistical weather downscaling, with this first release demonstrating strong generalization potential beyond the CONUS training domain.

---

## Overview

| | |
|---|---|
| **Input** | GraphCast 0.25°, 6-hourly global forecasts |
| **Output** | 1 km hourly surface fields, 67-hour window |
| **Domain** | CONUS (training); any global bounding box (inference) |
| **Architecture** | UNet3D + LCM scheduler |
| **Conditioning** | 20 channels: 17 GraphCast vars + elevation + SVF + cos-SZA |

### Output Variables

| Variable | Description | Unit |
|---|---|---|
| `APCP_surface` | Precipitation | mm/hr |
| `TMP_2maboveground` | 2-m Temperature | K |
| `SPFH_2maboveground` | 2-m Specific Humidity | kg/kg |
| `UGRD_10maboveground` | 10-m U-Wind | m/s |
| `VGRD_10maboveground` | 10-m V-Wind | m/s |
| `PRES_surface` | Surface Pressure | Pa |
| `DSWRF_surface` | Downward SW Radiation | W/m² |
| `DLWRF_surface` | Downward LW Radiation | W/m² |

---

## Repository Structure

```
earthmind_highres/
│
├── earthmind_highres_get_data.ipynb              # Data download and preprocessing
├── earthmind_highres_train_randomize_patches.py  # Patch index generation
│
├── em_train.py                   # UNet3D LCM training
├── em_inference.py               # Inference over any bounding box
│
├── em_evaluation.py              # CONUS evaluation vs AORC + HRRR + GraphCast
├── em_evaluation_global.py       # Global evaluation vs StationBench + GraphCast
│
├── LICENSE.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/shreesomnath/earthmind_highres.git
cd earthmind_highres
pip install torch xarray zarr fsspec diffusers numpy scipy matplotlib cartopy pyproj tqdm
```

> `cartopy` is optional but recommended for map plots with coastlines and borders.

---

## Usage

### 1. Data Preparation

Run `earthmind_highres_get_data.ipynb` to download and preprocess:
- AORC 1-km hourly surface fields
- GraphCast 6-hourly global forecasts
- Topography (normalized elevation) and sky-view factor static grids

### 2. Training

```bash
python em_train.py \
    --max-patches-per-ic 500 \
    --num-workers 4
```

| Argument | Default | Description |
|---|---|---|
| `--max-patches-per-ic` | `0` (all) | Max patches per IC; reduce for debugging |
| `--num-workers` | `0` | DataLoader workers for AORC loading |

Training resumes automatically from the last global-best checkpoint if one exists.

### 3. Inference

```bash
python em_inference.py \
    --graphcast_zarr  /path/to/graphcast.zarr \
    --gc_min_zarr     /path/to/gc_min.zarr \
    --gc_max_zarr     /path/to/gc_max.zarr \
    --topo_nc         /path/to/elevation.nc \
    --svf_nc          /path/to/sky_view_factor.nc \
    --aorc_min_nc     /path/to/aorc_min.nc \
    --aorc_max_nc     /path/to/aorc_max.nc \
    --checkpoint      /path/to/model.pt \
    --start_date 2021-07-04 \
    --end_date   2021-12-31 \
    --num_inits  10 \
    --lat_min 25.0 --lat_max 50.0 \
    --lon_min -125.0 --lon_max -65.0 \
    --patch 256 --stride 128 \
    --steps 4 8 25 50 \
    --out_dir /path/to/output \
    --out_name conus_inference \
    --overwrite
```

Each value in `--steps` produces a separate output zarr (e.g. `--steps 4 8 25 50` writes four files). Use `--gcs_zarr_root` instead of `--graphcast_zarr` for GCS-style directory layouts.

### 4. Evaluation — CONUS

Compares EarthMind-SR against AORC (ground truth), HRRR (NWP baseline), and GraphCast (AI baseline).

```bash
python em_evaluation.py \
    --earthmind_zarr      /path/to/earthmind_output.zarr \
    --graphcast_zarr_root /path/to/graphcast_root/ \
    --year 2021 \
    --save_dir /path/to/eval_output \
    --vars ALL \
    --lead_hours_sub6h 1 2 3 4 5 \
    --lead_hours_6h 6 12 18 24 36 48 \
    --lat_min 24.0 --lat_max 50.0 \
    --lon_min -125.0 --lon_max -66.0
```

### 5. Evaluation — Global (StationBench)

Compares EarthMind-SR against StationBench station observations (temperature only) and GraphCast.

```bash
python em_evaluation_global.py \
    --earthmind_zarr      /path/to/earthmind_output.zarr \
    --stationbench_zarr   /path/to/stationbench.zarr \
    --graphcast_zarr_root /path/to/graphcast_root/ \
    --save_dir /path/to/eval_output \
    --vars ALL \
    --lead_hours_6h 6 12 18 24 30 36 48 \
    --lat_min 6.0 --lat_max 36.0 \
    --lon_min 67.0 --lon_max 97.0
```

> StationBench ground-truth comparison is available for **temperature only**. All other variables are evaluated with PSD curves and spatial maps.

---

## Evaluation Outputs

```
eval_output/
├── metrics.csv              # r, RMSE, bias, MAE per variable / lead / IC
└── combined_plots/
    ├── TMP_2maboveground_IC...png    # Spatial map | Radial PSD | Scatter
    ├── APCP_surface_IC..._inst.png   # Instantaneous precipitation
    ├── APCP_surface_IC..._6hracc.png # 6-hr accumulated precipitation
    └── ...
```

Each figure uses a two-column Nature-quality layout: spatial maps (left) and radial PSD + scatter plots (right), saved at 400 DPI.

---

## Model Architecture

| Parameter | Value |
|---|---|
| Architecture | UNet3D + LCM scheduler |
| Conditioning channels | 20 (17 GraphCast vars + elevation + SVF + cos-SZA) |
| Output channels | 8 (AORC variables) |
| Block channels | 64 → 128 → 256 → 512 |
| Conditioning lead times (T_cond) | 12 (6 h → 72 h GraphCast steps) |
| Output lead times (T_target) | 67 (hourly, 0–66 h) |
| Training patch size | 64 × 64 px |
| Inference tile size | 256 × 256 px, stride 192 px |
| Tile blending | Cosine-taper spatial blending |
| Training period | 2021 (CONUS) |

---

## Citation

If you use EarthMind-SR in your research, please cite:

```bibtex
@article{luitel2026earthmindsr,
  title   = {{EarthMind-SR}: Kilometer-Scale Atmospheric Super-Resolution
             via Latent Consistency Diffusion},
  author  = {Luitel, Somnath and Singh, Manmeet},
  journal = {arXiv preprint arXiv:2026.XXXXX},
  year    = {2026}
}
```

---

## License

See `LICENSE.txt` for terms of use.

---

## Contact

**Somnath Luitel** and **Manmeet Singh**
AI Research Lab, Department of Earth, Environmental, and Atmospheric Sciences
Western Kentucky University, Bowling Green, KY, USA
