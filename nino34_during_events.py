#!/usr/bin/env python3
"""
Compute QBO(50 hPa) and Niño3.4 (detrended) 3-month means for ONE CMIP6 model,
then bootstrap monthly-composite means for four event categories and save results.

Output used in Fig 4 and supplementary Fig S8


Usage:
  python run_bootstrap_one_model.py CESM2-WACCM
  python run_bootstrap_one_model.py CESM2-WACCM --outdir ./bootstrap_out --nboot 10000 --ci 95

Output:
  <outdir>/<model>_qbo_enso_bootstrap_ci95_nboot10000.npz
"""

import os
import glob
import argparse

import numpy as np
import xarray as xr

# scipy is required for butter/filtfilt
from scipy.signal import butter, filtfilt


PLEV_HPA = 50
LON_MIN, LON_MAX = 50, 280
LAT_BAND = 10

UA_CHUNKS = {"time": 240}
SST_CHUNKS = {"time": 120}


# ---- INPUT PATHS (unchanged) ----
ua_patterns = {
    "AWI-CM-1-1-MR":    "/glade/collections/cmip/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/piControl/r1i1p1f1/Amon/ua/gn/v20181218/ua/*.nc",
    "BCC-CSM2-MR":      "/glade/collections/cmip/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Amon/ua/gn/v20181016/ua/*.nc",
    "CESM2-WACCM":      "/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2-WACCM/piControl/r1i1p1f1/Amon/ua/gn/v20190320/*.nc",
    "CNRM-CM6-1":       "/glade/collections/cmip/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/piControl/r1i1p1f2/Amon/ua/gr/v20180814/ua/*.nc",
    "CNRM-ESM2-1":      "/glade/collections/cmip/CMIP6/CMIP/CNRM-CERFACS/CNRM-ESM2-1/piControl/r1i1p1f2/Amon/ua/gr/v20181115/ua/*.nc",
    "E3SM-1-0":         "/glade/collections/cmip/CMIP6/CMIP/E3SM-Project/E3SM-1-0/piControl/r1i1p1f1/Amon/ua/gr/v20190723/ua/*.nc",
    "EC-Earth3":        "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/ua*.nc",
    "GFDL-ESM4":        "/glade/collections/cmip/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/piControl/r1i1p1f1/Amon/ua/gr1/v20180701/ua/*.nc",
    "HadGEM3-GC31-LL":  "/glade/collections/cmip/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Amon/ua/gn/v20190628/ua/*.nc",
    "IPSL-CM6A-LR":     "/glade/collections/cmip/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Amon/ua/gr/v20181123/ua/*.nc",
    "MIROC6":           "/glade/collections/cmip/CMIP6/CMIP/MIROC/MIROC6/piControl/r1i1p1f1/Amon/ua/gn/v20190311/ua/*.nc",
    "MPI-ESM1-2-HR":    "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/ua*.nc",
    "MRI-ESM2-0":       "/glade/collections/cmip/CMIP6/CMIP/MRI/MRI-ESM2-0/piControl/r1i1p1f1/Amon/ua/gn/v20190308/ua/*.nc",
    "UKESM1-0-LL":      "/glade/collections/cmip/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl/r1i1p1f2/Amon/ua/gn/v20190410/ua/*.nc",
}

tos_patterns = {
    "AWI-CM-1-1-MR":   "/glade/collections/cmip/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/piControl/r1i1p1f1/Omon/tos/gn/v20181218/tos/*.nc",
    "BCC-CSM2-MR":     "/glade/collections/cmip/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Omon/tos/gn/v20181015/tos/*.nc",
    "CESM2-WACCM":     "/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2-WACCM/piControl/r1i1p1f1/Omon/tos/gn/v20190320/*.nc",
    "CNRM-CM6-1":      "/glade/collections/cmip/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/piControl/r1i1p1f2/Omon/tos/gn/v20180814/tos/*.nc",
    "CNRM-ESM2-1":     "/glade/collections/cmip/CMIP6/CMIP/CNRM-CERFACS/CNRM-ESM2-1/piControl/r1i1p1f2/Omon/tos/gn/v20181115/tos/*.nc",
    "E3SM-1-0":        "/glade/collections/cmip/CMIP6/CMIP/E3SM-Project/E3SM-1-0/piControl/r1i1p1f1/Omon/tos/gr/v20191007/tos/*.nc",
    "EC-Earth3":       "/glade/collections/cmip/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/piControl/r1i1p1f1/Omon/tos/gn/v20190712/tos/*.nc",
    "GFDL-ESM4":       "/glade/collections/cmip/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/piControl/r1i1p1f1/Omon/tos/gn/v20180701/tos/*.nc",
    "HadGEM3-GC31-LL": "/glade/collections/cmip/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Omon/tos/gn/v20190628/tos/*.nc",
    "IPSL-CM6A-LR":    "/glade/collections/cmip/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Omon/tos/gn/v20181123/tos/*.nc",
    "MIROC6":          "/glade/collections/cmip/CMIP6/CMIP/MIROC/MIROC6/piControl/r1i1p1f1/Omon/tos/gn/v20181212/tos/*.nc",
    "MPI-ESM1-2-HR":   "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/tos*.nc",
    "MRI-ESM2-0":      "/glade/collections/cmip/CMIP6/CMIP/MRI/MRI-ESM2-0/piControl/r1i1p1f1/Omon/tos/gr/v20190904/tos/*.nc",
    "UKESM1-0-LL":     "/glade/collections/cmip/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl/r1i1p1f2/Omon/tos/gn/v20190827/tos/*.nc",
}


def qbo_index_from_cmip(pattern: str) -> xr.DataArray:
    ds = xr.open_mfdataset(
        sorted(glob.glob(pattern)),
        combine="by_coords",
        decode_times=True,
        use_cftime=True,
        chunks=UA_CHUNKS,
    )[["ua", "plev", "lat", "lon", "time"]]

    if (ds.lon.min() < 0).item():
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")

    sub = ds.sel(lon=slice(LON_MIN, LON_MAX), lat=slice(-LAT_BAND, LAT_BAND)).sortby("lat")

    ua50 = sub["ua"].sel(plev=PLEV_HPA * 100, method="nearest")
    ua50_zm = ua50.mean("lon")
    wlat = np.cos(np.deg2rad(ua50_zm.lat))
    qbo_raw = ua50_zm.weighted(wlat).mean("lat")

    clim = qbo_raw.groupby("time.month").mean("time")
    anom = qbo_raw.groupby("time.month") - clim
    fit = anom.polyfit(dim="time", deg=1)
    trend = xr.polyval(anom["time"], fit.polyfit_coefficients)
    qbo_detr = (anom - trend).astype("float32")

    qbo_detr = qbo_detr.load()  # small 1D array
    ds.close()
    return qbo_detr


def bandpass_filter_1d_numpy(da: xr.DataArray, fs=1.0, low_per=36, high_per=12, order=3) -> xr.DataArray:
    """Bandpass a 1D DataArray along 'time' using filtfilt."""
    f_low = 1.0 / low_per
    f_high = 1.0 / high_per
    nyquist = 0.5 * fs
    low_cut = f_low / nyquist
    high_cut = f_high / nyquist

    b, a = butter(order, [low_cut, high_cut], btype="band")
    arr = da.values.astype("float32")
    filt = filtfilt(b, a, arr, axis=0, padlen=0)

    return xr.DataArray(filt, coords=da.coords, dims=da.dims, name=da.name)


def three_month_composite(ts1d: xr.DataArray) -> xr.DataArray:
    return ts1d.rolling(time=3, center=True, min_periods=3).mean().dropna("time")


def compute_nino34_from_tos(tos_pattern: str) -> xr.DataArray:
    files = sorted(glob.glob(tos_pattern))
    if not files:
        raise FileNotFoundError(f"No tos files found for pattern: {tos_pattern}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        decode_times=True,
        use_cftime=True,
        chunks=SST_CHUNKS,
    )

    if "tos" not in ds:
        ds.close()
        raise KeyError("tos variable not found in dataset")
    da = ds["tos"]

    lat_candidates = [c for c in ds.coords if c.lower().startswith("lat")]
    lon_candidates = [c for c in ds.coords if c.lower().startswith("lon")]

    if not lat_candidates and "nav_lat" in ds.coords:
        lat_candidates = ["nav_lat"]
    if not lon_candidates and "nav_lon" in ds.coords:
        lon_candidates = ["nav_lon"]

    if not lat_candidates or not lon_candidates:
        ds.close()
        raise KeyError(f"Could not find lat/lon coords in SST dataset for {tos_pattern}")

    lat_name = lat_candidates[0]
    lon_name = lon_candidates[0]
    lat = ds[lat_name]
    lon = ds[lon_name]
    lon360 = lon % 360

    can_slice = (lat_name in da.dims) and (lon_name in da.dims)

    if can_slice:
        da2 = da.assign_coords({lon_name: lon360}).sortby(lon_name)

        # Ensure latitude monotonic increasing for slice (best effort)
        try:
            if (da2[lat_name][0] > da2[lat_name][-1]).compute():
                da2 = da2.sortby(lat_name)
        except Exception:
            if float(da2[lat_name].values[0]) > float(da2[lat_name].values[-1]):
                da2 = da2.sortby(lat_name)

        sub = da2.sel({lat_name: slice(-5, 5), lon_name: slice(190, 240)})
        wlat = np.cos(np.deg2rad(sub[lat_name]))
        n34_mean = sub.weighted(wlat).mean(dim=lat_name).mean(dim=lon_name)
    else:
        mask = (lat >= -5) & (lat <= 5) & (lon360 >= 190) & (lon360 <= 240)
        horiz_dims = tuple(d for d in mask.dims if d != "time")
        weights = np.cos(np.deg2rad(lat)).where(mask).fillna(0)

        n34_mean = da.where(mask).weighted(weights).mean(dim=horiz_dims)

    if "time" not in n34_mean.dims:
        ds.close()
        raise ValueError(f"No time dimension in Niño-3.4 mean for pattern {tos_pattern}")

    if not bool(n34_mean.notnull().any().values):
        ds.close()
        raise ValueError(f"No valid Niño-3.4 data for pattern {tos_pattern} (all NaN in box)")

    clim = n34_mean.groupby("time.month").mean("time")
    n34_anom = (n34_mean.groupby("time.month") - clim).rename("nino34")

    n34_anom = n34_anom.copy(deep=True)
    ds.close()
    return n34_anom


def detrend_1d(ts1d: xr.DataArray) -> xr.DataArray:
    fit = ts1d.polyfit(dim="time", deg=1)
    trend = xr.polyval(ts1d["time"], fit.polyfit_coefficients)
    return (ts1d - trend).astype("float32")


def bootstrap_mean_ci(data, n_boot=10000, ci=95, rng=None):
    """Returns (mean_of_data, lower_ci, upper_ci). If data empty -> (nan,nan,nan)."""
    data = np.asarray(data)
    if data.size == 0:
        return np.nan, np.nan, np.nan

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, data.size, size=(n_boot, data.size))
    boot_means = data[idx].mean(axis=1)

    mean_data = data.mean()
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return mean_data, lo, hi


def run_one_model(model: str, nboot: int, ci: float, qbo_threshold: float, seed: int | None):
    if model not in ua_patterns:
        raise KeyError(f"Model '{model}' not found in ua_patterns. Options: {sorted(ua_patterns)}")
    if model not in tos_patterns:
        raise KeyError(f"Model '{model}' not found in tos_patterns. Options: {sorted(tos_patterns)}")

    # --- QBO: ua at 50 hPa (deseasonalized & detrended) ---
    qbo = qbo_index_from_cmip(ua_patterns[model])
    qbo_bp = bandpass_filter_1d_numpy(qbo, low_per=36, high_per=12, order=3)
    qbo_3m = three_month_composite(qbo_bp)

    # --- Niño3.4: tos anomalies, then detrend ---
    n34 = compute_nino34_from_tos(tos_patterns[model])
    n34_dt = detrend_1d(n34)
    n34_3m = three_month_composite(n34_dt)

    # --- Align ---
    qbo_3m, n34_3m = xr.align(qbo_3m, n34_3m, join="inner")

    # ENSO threshold = 1 std for this model
    enso_threshold = float(n34_3m.std())

    # --- determine QBO threshold (model-relative "auto" if qbo_threshold <= 0) ---
    # If user passed <=0 treat as auto: 1 * std(qbo_3m). If that yields too few events,
    # fall back to 80th percentile of |qbo_3m|.
    if qbo_threshold <= 0:
        qbo_thr = float(qbo_3m.std())
        method = "sigma"
    else:
        qbo_thr = float(qbo_threshold)
        method = "fixed"

    # compute counts for diagnostics
    n_qbow = int((qbo_3m > qbo_thr).sum().item())
    n_qboe = int((qbo_3m < -qbo_thr).sum().item())

    # fallback if too few events
    min_events_total = 8  # tweakable: require at least this many total events across signs
    if (n_qbow + n_qboe) < min_events_total:
        qbo_thr_pct = float(np.percentile(np.abs(qbo_3m.values), 80))
        # only replace threshold if percentile is lower (i.e. more permissive) than current
        if qbo_thr_pct < qbo_thr:
            qbo_thr = qbo_thr_pct
            method = "percentile80_fallback"

        # recompute counts
        n_qbow = int((qbo_3m > qbo_thr).sum().item())
        n_qboe = int((qbo_3m < -qbo_thr).sum().item())

    # Final masks
    qboe_mask = qbo_3m < -qbo_thr
    qbow_mask = qbo_3m >  qbo_thr
    lanina_mask = n34_3m < -enso_threshold
    elnino_mask = n34_3m >  enso_threshold

    # Diagnostic prints
    print(f"Model {model}: qbo_threshold_used = {qbo_thr:.3f} (method={method})")
    print(f"  qbo_3m min/max = {float(qbo_3m.min()):.3f} / {float(qbo_3m.max()):.3f}")
    print(f"  n_qbow = {n_qbow}, n_qboe = {n_qboe} (total = {n_qbow+n_qboe})")
    print(f"  enso_threshold (1σ) = {enso_threshold:.4f}")

    center_months = np.arange(1, 13)
    rng = np.random.default_rng(seed)

    def monthly_boot(mask):
        means = np.zeros(12, dtype=float)
        lows  = np.zeros(12, dtype=float)
        highs = np.zeros(12, dtype=float)

        for i, m in enumerate(range(1, 13)):
            month_sel = (n34_3m["time"].dt.month == m)
            data = n34_3m.where(mask & month_sel).dropna("time").values
            means[i], lows[i], highs[i] = bootstrap_mean_ci(data, n_boot=nboot, ci=ci, rng=rng)
        return means, lows, highs

    mean_qbow, low_qbow, up_qbow       = monthly_boot(qbow_mask)
    mean_elnino, low_elnino, up_elnino = monthly_boot(elnino_mask)
    mean_qboe, low_qboe, up_qboe       = monthly_boot(qboe_mask)
    mean_lanina, low_lanina, up_lanina = monthly_boot(lanina_mask)

    results = {
        "model": model,
        "nboot": nboot,
        "ci": float(ci),
        "qbo_threshold": float(qbo_thr),
        "qbo_threshold_method": method,
        "enso_threshold": float(enso_threshold),
        "center_months": center_months,

        "mean_qbow": mean_qbow,
        "err_qbow_lower": mean_qbow - low_qbow,
        "err_qbow_upper": up_qbow - mean_qbow,

        "mean_elnino": mean_elnino,
        "err_elnino_lower": mean_elnino - low_elnino,
        "err_elnino_upper": up_elnino - mean_elnino,

        "mean_qboe": mean_qboe,
        "err_qboe_lower": mean_qboe - low_qboe,
        "err_qboe_upper": up_qboe - mean_qboe,

        "mean_lanina": mean_lanina,
        "err_lanina_lower": mean_lanina - low_lanina,
        "err_lanina_upper": up_lanina - mean_lanina,
    }
    return results


def save_npz(outpath: str, results: dict):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # np.savez wants flat arrays/scalars; keep strings as 0-d object
    np.savez(outpath, **{k: v for k, v in results.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="CMIP6 model name (key in ua_patterns/tos_patterns)")
    p.add_argument("--outdir", default="bootstrap_results", help="Output directory for .npz")
    p.add_argument("--nboot", type=int, default=10000, help="Bootstrap samples per month")
    p.add_argument("--ci", type=float, default=95.0, help="Confidence interval percent")
    p.add_argument("--qbo-threshold", type=float, default=5.0, help="QBO threshold in m/s for event definition")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = p.parse_args()

    results = run_one_model(
        model=args.model,
        nboot=args.nboot,
        ci=args.ci,
        qbo_threshold=args.qbo_threshold,
        seed=args.seed,
    )

    outname = f"{args.model}_qbo_enso_bootstrap_ci{int(args.ci)}_nboot{args.nboot}.npz"
    outpath = os.path.join(args.outdir, outname)
    save_npz(outpath, results)

    print(f"Saved: {outpath}")
    print(f"ENSO threshold (1σ of Niño3.4): {results['enso_threshold']:.4f}")


if __name__ == "__main__":
    main()