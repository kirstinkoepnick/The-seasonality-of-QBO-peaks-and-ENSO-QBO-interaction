#!/usr/bin/env python3
"""
Compute the true pressure-coordinate TEM Brewer-Dobson circulation for one CMIP6 model.

Run one model per batch job, e.g.
  python compute_cmip6_bdc_batch.py CESM2-WACCM --out-dir /glade/derecho/scratch/kkoepnick/qbo-enso/cmip/bdc_outputs

Outputs monthly TEM residual circulation fields by default:
  omega_star(time, plev, lat)      Pa s-1, positive downward
  bdc_upwelling(time, plev, lat)   hPa day-1, positive upward = -omega_star * 864
  v_star(time, plev, lat)          m s-1 residual meridional velocity
  tropical_bdc_index(time)         hPa day-1, cosine-weighted 20S-20N, 70-10 hPa mean

This is not an omega proxy. It uses the TEM residual pressure velocity:
  omega* = omega_bar + [1/(a cos phi)] d/dphi { cos phi * (v'theta')bar / theta_p }
  v*     = v_bar - d/dp { (v'theta')bar / theta_p }
where theta_p = d theta_bar / dp.
"""

import os
import glob
import argparse
import warnings
import numpy as np
import xarray as xr
from xarray.conventions import SerializationWarning

# -----------------------
# constants
A_EARTH = 6_371_000.0
RD = 287.05
CP = 1004.0
P0 = 100000.0
KAPPA = RD / CP

DEFAULT_CHUNKS = {"time": 365}

warnings.filterwarnings("ignore", category=SerializationWarning)

# -----------------------
# Same model/path structure as the EPF batch script.
# Edit paths here if you move files or add models.
flux_patterns_daily = {
    "CESM2-WACCM": {
        "ua":  "/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2-WACCM/piControl/r1i1p1f1/day/ua/gn/v20190320/*.nc",
        "wap": "/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2-WACCM/piControl/r1i1p1f1/day/wap/gn/v20190320/*.nc",
        "va":  "/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2-WACCM/piControl/r1i1p1f1/day/va/gn/v20190320/*.nc",
        "ta":  "/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2-WACCM/piControl/r1i1p1f1/day/ta/gn/v20190320/*.nc",
    },
    "CNRM-CM6-1": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-CM6-1/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-CM6-1/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-CM6-1/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-CM6-1/ta/day/*.nc",
    },
    "CNRM-ESM2-1": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-ESM2-1/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-ESM2-1/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-ESM2-1/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/CNRM-ESM2-1/ta/day/*.nc",
    },
    "EC-Earth3": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/ta/day/*.nc",
    },
    "GFDL-ESM4": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/GFDL-ESM4/ua/Eday/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/GFDL-ESM4/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/GFDL-ESM4/va/Eday/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/GFDL-ESM4/ta/Eday/*.nc",
    },
    "IPSL-CM6A-LR": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/IPSL-CM6A-LR/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/IPSL-CM6A-LR/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/IPSL-CM6A-LR/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/IPSL-CM6A-LR/ta/day/*.nc",
    },
    "MIROC6": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MIROC6/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MIROC6/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MIROC6/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MIROC6/ta/day/*.nc",
    },
    "MPI-ESM1-2-HR": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/ta/day/*.nc",
    },
    "MRI-ESM2-0": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MRI-ESM2-0/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MRI-ESM2-0/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MRI-ESM2-0/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MRI-ESM2-0/ta/day/*.nc",
    },
}


def _rename_common(ds):
    rename = {}
    for old, new in [("longitude", "lon"), ("latitude", "lat"), ("lev", "plev"), ("level", "plev")]:
        if old in ds.dims or old in ds.coords:
            rename[old] = new
    for old, new in [("U", "ua"), ("V", "va"), ("T", "ta"), ("OMEGA", "wap"), ("omega", "wap")]:
        if old in ds.data_vars:
            rename[old] = new
    return ds.rename(rename)


def _pressure_coord_pa(da):
    p = da["plev"]
    units = p.attrs.get("units", "").lower()
    if units in ["pa", "pascal", "pascals"]:
        p_pa = p
    elif units in ["hpa", "mb", "millibar", "millibars"]:
        p_pa = p * 100.0
    elif float(p.max()) > 2000.0:
        p_pa = p
    else:
        p_pa = p * 100.0
    p_pa = p_pa.astype("float64")
    p_pa.attrs["units"] = "Pa"
    return p_pa


def open_var(pattern, vname, chunks):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        decode_times=True,
        use_cftime=True,
        chunks=chunks,
        parallel=False,
        engine="netcdf4",
    )
    ds = _rename_common(ds)
    if "lon" in ds.coords and bool((ds.lon.min() < 0).item()):
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds[vname]


def subset_lat_lon_plev(da, lat_min, lat_max, lon_min, lon_max, pmin_hpa, pmax_hpa):
    da = da.sel(lat=slice(lat_min, lat_max))
    if lon_min is not None and lon_max is not None and lon_max < 360:
        da = da.sel(lon=slice(lon_min, lon_max))

    p_pa = _pressure_coord_pa(da)
    da = da.assign_coords(plev=p_pa).sortby("plev")
    pmin = pmin_hpa * 100.0
    pmax = pmax_hpa * 100.0
    da = da.sel(plev=slice(pmin, pmax))
    da["plev"].attrs["units"] = "Pa"
    return da


def d_dphi_lat(da):
    lat_deg = da["lat"].values
    lat_rad = np.deg2rad(lat_deg)
    tmp = da.assign_coords(lat=lat_rad)
    out = tmp.differentiate("lat")
    return out.assign_coords(lat=lat_deg)


def compute_tem_bdc(ua, va, wap, ta, theta_p_min=1e-6):
    """Return TEM residual v_star and omega_star from pressure-level data."""
    # Put all fields on the temperature grid and align times/levels exactly.
    lat, lon = ta.lat, ta.lon
    ua = ua.interp(lat=lat, lon=lon)
    va = va.interp(lat=lat, lon=lon)
    wap = wap.interp(lat=lat, lon=lon)
    ua, va, wap, ta = xr.align(ua, va, wap, ta, join="inner")

    # Make pressure Pa and ascending for derivatives.
    p = _pressure_coord_pa(ta)
    ua = ua.assign_coords(plev=p).sortby("plev")
    va = va.assign_coords(plev=p).sortby("plev")
    wap = wap.assign_coords(plev=p).sortby("plev")
    ta = ta.assign_coords(plev=p).sortby("plev")
    p = ta["plev"]

    theta = ta * (P0 / p) ** KAPPA
    theta.name = "theta"

    vbar = va.mean("lon", skipna=True)
    wbar = wap.mean("lon", skipna=True)
    thetabar = theta.mean("lon", skipna=True)

    vp = va - vbar
    thetap = theta - thetabar
    vtheta_bar = (vp * thetap).mean("lon", skipna=True)
    vtheta_bar.name = "vtheta_bar"

    theta_p = thetabar.differentiate("plev")
    theta_p.name = "theta_p"
    theta_p = theta_p.where(np.isfinite(theta_p))
    theta_p = theta_p.where(np.abs(theta_p) > theta_p_min)

    heat_flux_term = vtheta_bar / theta_p
    heat_flux_term.name = "vtheta_over_thetap"

    cosphi = xr.DataArray(
        np.cos(np.deg2rad(heat_flux_term["lat"].values)),
        coords={"lat": heat_flux_term["lat"].values},
        dims=("lat",),
        name="coslat",
    )

    omega_star = wbar + d_dphi_lat(cosphi * heat_flux_term) / (A_EARTH * cosphi)
    omega_star = omega_star.where(np.abs(omega_star["lat"]) <= 80)
    omega_star.name = "omega_star"
    omega_star.attrs.update({
        "units": "Pa s-1",
        "long_name": "TEM residual pressure velocity",
        "sign_convention": "positive downward; negative values are residual upwelling",
        "formula": "omega_bar + (a cosphi)^-1 d/dphi[cosphi * vtheta_bar / theta_p]",
    })

    v_star = vbar - heat_flux_term.differentiate("plev")
    v_star = v_star.where(np.abs(v_star["lat"]) <= 80)
    v_star.name = "v_star"
    v_star.attrs.update({
        "units": "m s-1",
        "long_name": "TEM residual meridional velocity",
        "formula": "v_bar - d/dp[vtheta_bar / theta_p]",
    })

    bdc_upwelling = -omega_star * 864.0
    bdc_upwelling.name = "bdc_upwelling"
    bdc_upwelling.attrs.update({
        "units": "hPa day-1",
        "long_name": "TEM BDC residual upwelling, -omega_star",
        "sign_convention": "positive upward",
        "conversion": "-omega_star[Pa s-1] * 864 = hPa day-1",
    })

    out = xr.Dataset({
        "omega_star": omega_star,
        "bdc_upwelling": bdc_upwelling,
        "v_star": v_star,
        "vtheta_bar": vtheta_bar,
        "theta_p": theta_p,
        "heat_flux_term": heat_flux_term,
    })
    out["plev"].attrs["units"] = "Pa"
    return out


def add_tropical_index(ds, lat_bounds=(-20.0, 20.0), p_bounds_hpa=(70.0, 10.0)):
    p_bot = max(p_bounds_hpa) * 100.0
    p_top = min(p_bounds_hpa) * 100.0
    bdc = ds["bdc_upwelling"].sel(lat=slice(lat_bounds[0], lat_bounds[1]))
    bdc = bdc.sel(plev=slice(p_top, p_bot))
    bdc = bdc.where(np.isfinite(bdc))
    bdc = bdc.where(np.abs(bdc) < 2.0)
    weights = xr.DataArray(
        np.cos(np.deg2rad(bdc["lat"].values)),
        coords={"lat": bdc["lat"].values},
        dims=("lat",),
    )
    idx = bdc.weighted(weights).mean(("lat", "plev"), skipna=True)
    idx.name = "tropical_bdc_index"
    idx.attrs.update({
        "units": "hPa day-1",
        "long_name": "Tropical TEM BDC upwelling index",
        "description": f"-omega_star averaged over {lat_bounds[0]} to {lat_bounds[1]} deg and {max(p_bounds_hpa)}-{min(p_bounds_hpa)} hPa. Positive upward.",
    })
    ds["tropical_bdc_index"] = idx
    return ds


def maybe_resample_monthly(ds, output_frequency):
    if output_frequency == "daily":
        return ds
    if output_frequency == "monthly":
        return ds.resample(time="MS").mean("time", skipna=True)
    if output_frequency == "monthly_climatology":
        return ds.groupby("time.month").mean("time", skipna=True)
    raise ValueError(f"Unknown output frequency: {output_frequency}")


def run_one_model(model_name, args):
    if model_name not in flux_patterns_daily:
        raise KeyError(f"Model {model_name!r} not configured. Options: {list(flux_patterns_daily)}")

    paths = flux_patterns_daily[model_name]
    chunks = {"time": args.time_chunk}

    print(f"[{model_name}] opening variables")
    ua = open_var(paths["ua"], "ua", chunks)
    va = open_var(paths["va"], "va", chunks)
    wap = open_var(paths["wap"], "wap", chunks)
    ta = open_var(paths["ta"], "ta", chunks)

    print(f"[{model_name}] subsetting domain")
    ua = subset_lat_lon_plev(ua, args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.pmin_hpa, args.pmax_hpa)
    va = subset_lat_lon_plev(va, args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.pmin_hpa, args.pmax_hpa)
    wap = subset_lat_lon_plev(wap, args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.pmin_hpa, args.pmax_hpa)
    ta = subset_lat_lon_plev(ta, args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.pmin_hpa, args.pmax_hpa)

    print(f"[{model_name}] computing TEM BDC lazily")
    ds_bdc = compute_tem_bdc(ua, va, wap, ta, theta_p_min=args.theta_p_min)
    ds_bdc = maybe_resample_monthly(ds_bdc, args.output_frequency)
    ds_bdc = add_tropical_index(ds_bdc, lat_bounds=(args.index_lat_min, args.index_lat_max), p_bounds_hpa=(args.index_pbot_hpa, args.index_ptop_hpa))

    ds_bdc.attrs.update({
        "model": model_name,
        "description": "Pressure-coordinate TEM Brewer-Dobson circulation from daily CMIP6 fields; true residual circulation, not an omega proxy.",
        "source_frequency": "daily",
        "output_frequency": args.output_frequency,
        "lat_subset_deg": f"{args.lat_min} to {args.lat_max}",
        "pressure_subset_hpa": f"{args.pmin_hpa} to {args.pmax_hpa}",
    })

    if args.keep_last_years is not None and "time" in ds_bdc.dims:
        ntime = int(args.keep_last_years * (365 if args.output_frequency == "daily" else 12))
        ds_bdc = ds_bdc.isel(time=slice(-ntime, None))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{model_name}_TEM_BDC_{args.output_frequency}.nc")
    tmp_path = out_path + ".tmp"

    encoding = {}
    for v in ds_bdc.data_vars:
        if args.complevel > 0:
            encoding[v] = {"zlib": True, "complevel": args.complevel}
        else:
            encoding[v] = {}

    print(f"[{model_name}] writing {out_path}")
    ds_bdc.to_netcdf(tmp_path, encoding=encoding)
    os.replace(tmp_path, out_path)
    print(f"[{model_name}] done")


def parse_args():
    p = argparse.ArgumentParser(description="Compute one-model-at-a-time true TEM BDC for CMIP6 models used in the EPF script.")
    p.add_argument("model", help=f"Model name, or ALL. Options: {', '.join(flux_patterns_daily)}")
    p.add_argument("--out-dir", default="/glade/derecho/scratch/kkoepnick/qbo-enso/cmip")
    p.add_argument("--lat-min", type=float, default=-20.0)
    p.add_argument("--lat-max", type=float, default=20.0)
    p.add_argument("--lon-min", type=float, default=0.0)
    p.add_argument("--lon-max", type=float, default=360.0)
    p.add_argument("--pmin-hpa", type=float, default=1.0, help="top pressure bound, hPa")
    p.add_argument("--pmax-hpa", type=float, default=100.0, help="bottom pressure bound, hPa")
    p.add_argument("--index-lat-min", type=float, default=-20.0)
    p.add_argument("--index-lat-max", type=float, default=20.0)
    p.add_argument("--index-pbot-hpa", type=float, default=70.0)
    p.add_argument("--index-ptop-hpa", type=float, default=10.0)
    p.add_argument("--output-frequency", choices=["daily", "monthly", "monthly_climatology"], default="monthly")
    p.add_argument("--keep-last-years", type=int, default=None, help="Optionally keep only the last N years after computing/resampling")
    p.add_argument("--time-chunk", type=int, default=365)
    p.add_argument("--theta-p-min", type=float, default=1e-6)
    p.add_argument("--complevel", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.model == "ALL":
        # Mainly for interactive testing. For batch jobs, submit one model per job instead.
        for model_name in flux_patterns_daily:
            run_one_model(model_name, args)
    else:
        run_one_model(args.model, args)


if __name__ == "__main__":
    main()