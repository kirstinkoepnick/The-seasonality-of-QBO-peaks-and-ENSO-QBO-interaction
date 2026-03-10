#!/usr/bin/env python3
import os
import glob
import argparse
import warnings
import numpy as np
import xarray as xr
from xarray.conventions import SerializationWarning

# -----------------------
# constants
a = 6_371_000.0
Omega = 7.292115e-5
Rd = 287.05
Cp = 1004.0
P0 = 100000.0
H = 7000.0  # scale height for log-pressure height z* = -H ln(p/P0)

CHUNKS = {"time": 365}

# -----------------------
# silence serialization warnings about multiple fill values
warnings.filterwarnings("ignore", category=SerializationWarning, message="variable 'va' has multiple fill values")
warnings.filterwarnings("ignore", category=SerializationWarning, message="variable 'ta' has multiple fill values")
warnings.filterwarnings("ignore", category=SerializationWarning, message="variable 'wap' has multiple fill values")
warnings.filterwarnings("ignore", category=SerializationWarning, message="variable 'ua' has multiple fill values")


def open_var(pattern, vname):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        decode_times=True,
        use_cftime=True,
        chunks=CHUNKS,
        parallel=False,
        engine="netcdf4",
    )

    # normalize lon to 0..360
    if "lon" in ds.coords:
        if (ds.lon.min() < 0).item():
            ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")

    return ds[vname]


def subset_lat_lon(da, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    if (lat_min is not None) and (lat_max is not None):
        da = da.sel(lat=slice(lat_min, lat_max))
    if (lon_min is not None) and (lon_max is not None):
        da = da.sel(lon=slice(lon_min, lon_max))
    return da


def compute_Fz_latmean_TEM_from_cmip6(ua, va, wap, ta):
    """
    Compute TEM EP-flux vertical component Fz following Kim & Chun (2015) Eq. (3),
    consistently in log-pressure height.

    Inputs:
      ua  : zonal wind (m/s)
      va  : meridional wind (m/s)
      wap : omega (Pa/s)
      ta  : temperature (K)

    Returns:
      Fz_latmean : (time, plev) Cos(phi)-weighted latitude mean EP flux vertical component in kg/s^2
    """
    lat, lon, p = ta.lat, ta.lon, ta.plev

    ua = ua.interp(lat=lat, lon=lon)
    va = va.interp(lat=lat, lon=lon)
    om = wap.interp(lat=lat, lon=lon)

    ua, va, om, ta = xr.align(ua, va, om, ta, join="inner")

    # Ensure monotonic plev for vertical derivatives
    if not (np.all(np.diff(p.values) > 0) or np.all(np.diff(p.values) < 0)):
        ua = ua.sortby("plev")
        va = va.sortby("plev")
        om = om.sortby("plev")
        ta = ta.sortby("plev")
        p = ta.plev

    latr = np.deg2rad(lat)
    cosphi = xr.DataArray(np.cos(latr), dims=["lat"], coords={"lat": lat})
    f = xr.DataArray(2.0 * Omega * np.sin(latr), dims=["lat"], coords={"lat": lat})

    theta = ta * (P0 / p) ** (Rd / Cp)

    # zonal means
    ubar = ua.mean("lon")
    thbar = theta.mean("lon")

    # eddies
    up  = ua - ubar
    vp  = va - va.mean("lon")
    thp = theta - thbar

    vpthp_bar = (vp * thp).mean("lon")   # \overline{v' theta'}

    # density from zonal-mean T
    Tbar = ta.mean("lon")
    p3d = xr.DataArray(p, dims=["plev"], coords={"plev": p})
    p3d, _ = xr.broadcast(p3d, Tbar)      # (time,plev,lat)
    rho0 = p3d / (Rd * Tbar)

    # log-p height z* = -H ln(p/P0)
    zstar = xr.DataArray(-H * np.log(p / P0), dims=["plev"], coords={"plev": p})

    # d\bar{theta}/dz*
    thbar_z = thbar.differentiate("plev") / zstar.differentiate("plev")
    thbar_z = thbar_z.where(np.abs(thbar_z) > 1e-10)

    # zdot = dz*/dt = -H * omega / p
    p4d = xr.DataArray(p, dims=["plev"], coords={"plev": p})
    p4d, _ = xr.broadcast(p4d, om)        # (time,plev,lat,lon)
    zdot = -H * om / p4d

    zdotp = zdot - zdot.mean("lon")
    zdotup_bar = (zdotp * up).mean("lon")  # \overline{w' u'} in z* coord

    # bracket term: f - (a cosphi)^-1 (ubar cosphi)_phi
    phi = xr.DataArray(latr, dims=["lat"], coords={"lat": lat})
    dphi_dlat = phi.differentiate("lat")
    dudphi_term = (ubar * cosphi).differentiate("lat") / dphi_dlat
    bracket = f - (1.0 / (a * cosphi)) * dudphi_term

    Fz = rho0 * a * cosphi * (bracket * (vpthp_bar / thbar_z) - zdotup_bar)
    Fz.name = "Fz"

    Fz_latmean = Fz.weighted(cosphi).mean("lat")
    Fz_latmean.name = "Fz_latmean"  # kg/s^2
    return Fz_latmean


# -----------------------
# file patterns by model
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
    # NOTE: you had these mislabeled in your paste; fix to real paths if needed.
    "EC-Earth3": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/ua/day/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/wap/day/*.nc",
        "va":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/va/day/*.nc",
        "ta":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/ta/day/*.nc",
    },
    "GFDL-ESM4": {
        "ua":  "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/GFDL-ESM4/ua/Eday/*.nc",
        "wap": "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/GFDL-ESM4/wap/day/*.nc",  # edit if needed
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


def run_one_model(model_name: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float, out_dir: str):
    if model_name not in flux_patterns_daily:
        raise KeyError(f"Model '{model_name}' not in flux_patterns_daily. Options: {list(flux_patterns_daily)}")

    paths = flux_patterns_daily[model_name]

    print(f"[{model_name}] opening variables...")
    ua  = open_var(paths["ua"],  "ua")
    va  = open_var(paths["va"],  "va")
    wap = open_var(paths["wap"], "wap")
    ta  = open_var(paths["ta"],  "ta")

    print(f"[{model_name}] subsetting lat/lon...")
    ua  = subset_lat_lon(ua,  lat_min, lat_max, lon_min, lon_max)
    va  = subset_lat_lon(va,  lat_min, lat_max, lon_min, lon_max)
    wap = subset_lat_lon(wap, lat_min, lat_max, lon_min, lon_max)
    ta  = subset_lat_lon(ta,  lat_min, lat_max, lon_min, lon_max)

    print(f"[{model_name}] computing Fz_latmean (lazy)...")
    Fz_latmean = compute_Fz_latmean_TEM_from_cmip6(ua, va, wap, ta)

    print(f"[{model_name}] materializing with .compute()...")
    Fz_latmean = Fz_latmean.compute()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_Fz_latmean_daily.nc")

    print(f"[{model_name}] saving to: {out_path}")
    Fz_latmean.to_netcdf(out_path)
    print(f"[{model_name}] done.")


def main():
    parser = argparse.ArgumentParser(description="Compute daily TEM EP-flux vertical component Fz_latmean and save to NetCDF.")
    parser.add_argument("model", help=f"Model name. Options: {', '.join(flux_patterns_daily.keys())}")
    parser.add_argument("--lat-min", type=float, default=-5.0)
    parser.add_argument("--lat-max", type=float, default=5.0)
    parser.add_argument("--lon-min", type=float, default=0.0)
    parser.add_argument("--lon-max", type=float, default=360.0)
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    run_one_model(
        model_name=args.model,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()