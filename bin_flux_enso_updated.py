#!/usr/bin/env python3
"""
Bin corrected EP flux (Fz_latmean) by ENSO (Niño-3.4) for multiple CMIP6 models.

Key points:
- Flux is daily -> monthly mean (MS) before binning.
- Niño-3.4 is computed from CMIP6 Omon/tos with cftime time (often mid-month).
- We align flux and Niño by a year-month key ('ym') to avoid month-start vs mid-month mismatches.
- We avoid carrying groupby-generated coords like 'month' in the Niño index.
- Output is computed into memory before pickling (safe for multiprocessing).
"""

import os
import glob
import pickle
import multiprocessing as mp

import numpy as np
import xarray as xr


# --------------------------
# User settings / inputs
# --------------------------
BINS = np.arange(-2.6, 2.6 + 0.8, 0.8)
SST_CHUNKS = {"time": 120}

# NEW: corrected EP flux input
DATA_DIR = "/glade/u/home/kkoepnick/qbo-enso/saved_data/"
MODEL_NAMES_FZ = [
    "CESM2-WACCM", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3",
    "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"
]
FLUX_VAR = "Fz_latmean"
FLUX_FILE_TEMPLATE = "{model}_Fz_latmean_daily.nc"

OUT_PICKLE = "/glade/u/home/kkoepnick/qbo-enso/saved_data/flux_binned_by_model_epf_corrected.pkl"

TOS_PATTERNS = {
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
    "MPI-ESM1-2-HR":   "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/tos/Omon/tos*.nc",
    "MRI-ESM2-0":      "/glade/collections/cmip/CMIP6/CMIP/MRI/MRI-ESM2-0/piControl/r1i1p1f1/Omon/tos/gr/v20190904/tos/*.nc",
    "UKESM1-0-LL":     "/glade/collections/cmip/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl/r1i1p1f2/Omon/tos/gn/v20190827/tos/*.nc",
}


# --------------------------
# Helpers
# --------------------------
def _drop_non_time_coords(da: xr.DataArray) -> xr.DataArray:
    """Drop non-time coords (e.g., groupby-generated 'month') and return in-memory copy."""
    da = da.rename("nino34")
    for c in list(da.coords):
        if c != "time" and "time" not in da[c].dims:
            da = da.drop_vars(c)
    return da.copy(deep=True)


def _to_inmem(da: xr.DataArray) -> xr.DataArray:
    """Compute into memory but preserve all coords/dims."""
    return da.compute().copy(deep=True)


def bin_flux_by_nino(flux_mon: xr.DataArray, n34_anom: xr.DataArray, bins: np.ndarray) -> xr.DataArray:
    """
    Bin monthly flux anomalies by Niño-3.4 anomalies.

    flux_mon: DataArray with dim 'time' (monthly)
    n34_anom: DataArray with dim 'time' (monthly-ish; may be mid-month cftime)
    Returns: DataArray with dims (plev, nino34_bins) or (nino34_bins, ...) depending on flux dims.
    """
    ef_anom = flux_mon - flux_mon.mean("time")
    n34 = n34_anom.rename("nino34")

    # align by year-month key to avoid month-start vs mid-month mismatch
    ef_ym = ef_anom.assign_coords(ym=ef_anom.time.dt.strftime("%Y-%m")).swap_dims({"time": "ym"})
    n34_ym = n34.assign_coords(ym=n34.time.dt.strftime("%Y-%m")).swap_dims({"time": "ym"})

    ef_aligned, n34_aligned = xr.align(ef_ym, n34_ym, join="inner")
    if ef_aligned.sizes["ym"] == 0:
        raise ValueError("No overlapping months between flux and nino34 after alignment")

    # groupby_bins needs a 1D, NumPy-backed grouping variable
    n34_group = n34_aligned.load()
    gb = ef_aligned.groupby_bins(n34_group, bins=bins, right=False)

    flux_binned = gb.mean("ym")

    # drop empty bins (use a robust reduction over all non-bin dims)
    counts = gb.count("ym").compute()
    nonbin_dims = [d for d in counts.dims if d != "nino34_bins"]
    if nonbin_dims:
        counts_1d = counts.isel({d: 0 for d in nonbin_dims})
    else:
        counts_1d = counts

    valid_bins = flux_binned.nino34_bins.values[(counts_1d > 0).values]
    flux_binned = flux_binned.sel(nino34_bins=valid_bins)

    return flux_binned


def compute_nino34_from_tos(tos_pattern: str) -> xr.DataArray:
    """
    Compute Niño-3.4 anomalies (monthly climatology removed) from CMIP6 Omon/tos files.
    Handles both rectilinear and curvilinear grids.
    """
    files = sorted(glob.glob(tos_pattern))
    if not files:
        raise FileNotFoundError(f"No tos files found for pattern: {tos_pattern}")

    with xr.open_mfdataset(
        files,
        combine="by_coords",
        decode_times=True,
        use_cftime=True,
        chunks=SST_CHUNKS,
    ) as ds:
        da = ds["tos"]

        lat_candidates = [c for c in ds.coords if c.lower().startswith("lat")]
        lon_candidates = [c for c in ds.coords if c.lower().startswith("lon")]

        if not lat_candidates and "nav_lat" in ds.coords:
            lat_candidates = ["nav_lat"]
        if not lon_candidates and "nav_lon" in ds.coords:
            lon_candidates = ["nav_lon"]

        if not lat_candidates or not lon_candidates:
            raise KeyError(f"Could not find lat/lon coords in SST dataset for {tos_pattern}")

        lat_name = lat_candidates[0]
        lon_name = lon_candidates[0]

        lat = ds[lat_name]
        lon = ds[lon_name]
        lon360 = lon % 360

        can_slice = (lat_name in da.dims) and (lon_name in da.dims)

        if can_slice:
            da2 = da.assign_coords({lon_name: lon360}).sortby(lon_name)

            # ensure latitude increasing
            if (da2[lat_name][0] > da2[lat_name][-1]).compute():
                da2 = da2.sortby(lat_name)

            sub = da2.sel({lat_name: slice(-5, 5), lon_name: slice(190, 240)})

            wlat = np.cos(np.deg2rad(sub[lat_name]))
            n34_mean = sub.weighted(wlat).mean(dim=lat_name).mean(dim=lon_name)
        else:
            mask = ((lat >= -5) & (lat <= 5) & (lon360 >= 190) & (lon360 <= 240))
            horiz_dims = tuple(d for d in mask.dims if d != "time")
            weights = np.cos(np.deg2rad(lat)).where(mask).fillna(0)
            n34_mean = da.where(mask).weighted(weights).mean(dim=horiz_dims)

        clim = n34_mean.groupby("time.month").mean("time")
        n34_anom = (n34_mean.groupby("time.month") - clim).rename("nino34")

        return _drop_non_time_coords(n34_anom)


def load_corrected_flux_monthly(model: str) -> xr.DataArray:
    """
    Load corrected daily EP flux for one model and convert to monthly (MS) means.
    Expects file: {DATA_DIR}/{model}_Fz_latmean_daily.nc
    Variable: Fz_latmean
    """
    fn = os.path.join(DATA_DIR, FLUX_FILE_TEMPLATE.format(model=model))
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing corrected EP flux file for {model}: {fn}")

    ds = xr.open_dataset(fn, engine="netcdf4")
    if FLUX_VAR not in ds:
        raise KeyError(f"{fn} does not contain variable '{FLUX_VAR}'. Variables: {list(ds.data_vars)}")

    da = ds[FLUX_VAR]

    # daily -> monthly, keep time axis
    flux_mon = da.resample(time="MS").mean()

    # (optional) close file handle, but keep data lazy
    ds.close()
    return flux_mon


def _work_one(model: str):
    print(f"Binning corrected EP flux by ENSO for {model} ...", flush=True)

    if model not in TOS_PATTERNS:
        raise KeyError(f"{model} not in TOS_PATTERNS")

    flux_mon = load_corrected_flux_monthly(model)
    n34_anom = compute_nino34_from_tos(TOS_PATTERNS[model])

    flux_binned = bin_flux_by_nino(flux_mon, n34_anom, bins=BINS)

    return model, _to_inmem(flux_binned)


# --------------------------
# Main
# --------------------------
def main():
    items = list(MODEL_NAMES_FZ)

    nprocs = 8
    ctx = mp.get_context("forkserver")
    chunksize = max(1, len(items) // (nprocs * 4))

    flux_binned_by_model = {}
    with ctx.Pool(processes=nprocs) as pool:
        for name, flux_binned in pool.imap_unordered(_work_one, items, chunksize=chunksize):
            flux_binned_by_model[name] = flux_binned

    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(flux_binned_by_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved: {OUT_PICKLE}")


if __name__ == "__main__":
    main()