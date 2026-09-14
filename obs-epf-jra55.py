import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy.signal import welch
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
from pathlib import Path
import pandas as pd
import os
import sys
# --- helpers ---------------------------------------------------
def norm_time(ds):
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds

def wrap_lon(ds):
    """
    Wrap longitude to 0–360 regardless of whether the coord is named
    'lon' or 'longitude'.
    """
    if "longitude" in ds.coords:
        lon_name = "longitude"
    elif "lon" in ds.coords:
        lon_name = "lon"
    else:
        raise KeyError(f"No longitude coordinate found. Coords: {list(ds.coords)}")

    if float(ds[lon_name].max()) <= 180:
        ds = ds.assign_coords({lon_name: ds[lon_name] % 360}).sortby(lon_name)

    return ds

jra_dir = Path("/glade/campaign/collections/rda/data/d628000/anl_p125")

year = int(sys.argv[1])

start = f"{year}-01-01"
end   = f"{year}-12-31"

cache_dir = Path("/glade/derecho/scratch/kkoepnick/eccodes_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["ECCODES_CACHE_DIR"] = str(cache_dir)
os.environ["ECCODES_CACHE"] = "1"

def get_jra_files(code, short_name):
    return [
        str(p)
        for y in range(pd.to_datetime(start).year, pd.to_datetime(end).year + 1)
        for p in sorted((jra_dir / f"{y}").glob(f"anl_p125.{code}_{short_name}.*"))
    ]

def load_jra55_daily(code, short_name, varname):
    files = get_jra_files(code, short_name)

    print(f"loading {short_name}: {len(files)} files")
    if not files:
        raise FileNotFoundError(f"No files found for {code}_{short_name}")

    ds = xr.open_mfdataset(
        files,
        engine="cfgrib",
        combine="by_coords",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
        chunks={"time": 120},
        parallel=False,
    )

    ds = wrap_lon(norm_time(ds))

    lev_name = "isobaricInhPa"
    if lev_name not in ds.coords:
        raise KeyError(f"Expected pressure coord '{lev_name}' not found. Coords: {list(ds.coords)}")

    da = ds[varname].sel(latitude=slice(5, -5), time=slice(start, end),)

    da = da.resample(time="1D").mean()

    # rename to match your existing EP flux function
    da = da.rename({
        "isobaricInhPa": "level",
    })

    return da.chunk({"time": 365})
print("loading variables...")
u_jra55 = load_jra55_daily("033", "ugrd", "u")
v_jra55 = load_jra55_daily("034", "vgrd", "v")
w_jra55 = load_jra55_daily("039", "vvel", "w")
t_jra55 = load_jra55_daily("011", "tmp",  "t")

def compute_Fz_latmean(
    u, v, omega, T,
    lat_bounds=(-5, 5),
    lon_dim="longitude",
    lat_dim="latitude",
    p_dim="level",
):
    a = 6.371e6
    Omega = 7.292115e-5
    Rd = 287.05
    Cp = 1004.0
    P0 = 100000.0
    H = 7000.0

    p = u[p_dim]
    p_pa = p * 100.0 if float(p.max()) < 2000 else p

    u = u.assign_coords({p_dim: p_pa})
    v = v.assign_coords({p_dim: p_pa})
    omega = omega.assign_coords({p_dim: p_pa})
    T = T.assign_coords({p_dim: p_pa})

    if not np.all(np.diff(p_pa.values) > 0):
        u = u.sortby(p_dim)
        v = v.sortby(p_dim)
        omega = omega.sortby(p_dim)
        T = T.sortby(p_dim)
    p_pa = u[p_dim]

    latr = np.deg2rad(u[lat_dim])
    cosphi = xr.DataArray(np.cos(latr), dims=[lat_dim], coords={lat_dim: u[lat_dim]})
    f = xr.DataArray(2.0 * Omega * np.sin(latr), dims=[lat_dim], coords={lat_dim: u[lat_dim]})

    theta = T * (P0 / p_pa) ** (Rd / Cp)

    ubar = u.mean(lon_dim)
    thbar = theta.mean(lon_dim)

    up = u - ubar
    vp = v - v.mean(lon_dim)
    thp = theta - thbar

    vpthp_bar = (vp * thp).mean(lon_dim)

    zstar = xr.DataArray(
        -H * np.log(p_pa / P0),
        dims=[p_dim],
        coords={p_dim: p_pa},
    )

    thbar_z = thbar.differentiate(p_dim) / zstar.differentiate(p_dim)
    thbar_z = thbar_z.where(np.abs(thbar_z) > 1e-10)

    zdot = -H * omega / p_pa
    zdotp = zdot - zdot.mean(lon_dim)
    zdotup_bar = (zdotp * up).mean(lon_dim)

    Tbar = T.mean(lon_dim)
    rho0 = p_pa / (Rd * Tbar)

    phi = xr.DataArray(latr, dims=[lat_dim], coords={lat_dim: u[lat_dim]})
    dphi_dlat = phi.differentiate(lat_dim)
    dudphi_term = (ubar * cosphi).differentiate(lat_dim) / dphi_dlat

    bracket = f - (1.0 / (a * cosphi)) * dudphi_term

    Fz = rho0 * a * cosphi * (
        bracket * (vpthp_bar / thbar_z) - zdotup_bar
    )

    lat_min, lat_max = sorted(lat_bounds)
    if Fz[lat_dim][0] > Fz[lat_dim][-1]:
        Fz = Fz.sel({lat_dim: slice(lat_max, lat_min)})
    else:
        Fz = Fz.sel({lat_dim: slice(lat_min, lat_max)})

    weights = np.cos(np.deg2rad(Fz[lat_dim]))
    Fz_latmean = Fz.weighted(weights).mean(lat_dim)

    Fz_latmean.name = "Fz"
    Fz_latmean.attrs["long_name"] = "Vertical EP flux component in log-pressure height coordinates"
    Fz_latmean.attrs["units"] = "kg s-2"

    return Fz_latmean
print("calculating EPF_z...")
Fz_jra55 = compute_Fz_latmean(
    u_jra55, v_jra55, w_jra55, t_jra55,
    lat_bounds=(-5, 5),
    lon_dim="longitude",
    lat_dim="latitude",
    p_dim="level",
)
print("monthly...")
Fz_jra55_monthly = Fz_jra55.resample(time="MS").mean()
Fz_jra55_monthly.name = "Fz"
print("loading into memory...")
Fz_jra55_monthly = Fz_jra55_monthly.load()
Fz_jra55_monthly.attrs["long_name"] = "Monthly mean vertical EP flux component in log-pressure height coordinates"
Fz_jra55_monthly.attrs["frequency"] = "monthly mean from daily JRA-55 values"
Fz_jra55_monthly.attrs["units"] = "kg s-2"
print("saving...")
outfile = f"/glade/derecho/scratch/kkoepnick/jra55_daily/Fz_jra55_{year}.nc"

Fz_jra55_monthly.to_netcdf(outfile)
print("done :)")