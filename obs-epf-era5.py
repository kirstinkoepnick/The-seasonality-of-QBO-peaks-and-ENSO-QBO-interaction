import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy.signal import welch
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
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

pl_dir = Path("/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl")

year = int(sys.argv[1])

start = f"{year}-01-01"
end   = f"{year}-12-31"

def yyyymm_range(start, end):
    months = pd.date_range(
        pd.to_datetime(start).replace(day=1),
        pd.to_datetime(end).replace(day=1),
        freq="MS"
    )
    return [d.strftime("%Y%m") for d in months]

months = yyyymm_range(start, end)

def get_files(param_code, short_name):
    return [
        str(p)
        for ym in months
        for p in sorted((pl_dir / ym).glob(f"*128_{param_code}_{short_name}*.nc"))
    ]

def make_preprocess(varname):
    def _preprocess(ds):
        ds = norm_time(ds)
        ds = ds.sel(latitude=slice(5, -5))
        return ds[[varname]]
    return _preprocess

def load_era5_daily_fast(param_code, short_name, varname):
    files = get_files(param_code, short_name)

    print(f"loading {short_name}: {len(files)} files")
    if not files:
        raise FileNotFoundError(f"No files found for 128_{param_code}_{short_name}")

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        preprocess=make_preprocess(varname),
        chunks={"time": 24},
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    ds = wrap_lon(ds)

    da = ds[varname].sel(time=slice(start, end))

    # daily mean after spatial/level subset
    da = da.resample(time="1D").mean()

    return da.chunk({"time": 365})

print("loading variables...")
u_era5 = load_era5_daily_fast("131", "u", "U")
v_era5 = load_era5_daily_fast("132", "v", "V")
w_era5 = load_era5_daily_fast("135", "w", "W")
t_era5 = load_era5_daily_fast("130", "t", "T")

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

    # pressure in Pa
    p = u[p_dim]
    p_pa = p * 100.0 if float(p.max()) < 2000 else p

    # align pressure coordinate consistently
    u = u.assign_coords({p_dim: p_pa})
    v = v.assign_coords({p_dim: p_pa})
    omega = omega.assign_coords({p_dim: p_pa})
    T = T.assign_coords({p_dim: p_pa})

    # sort pressure if needed
    if not np.all(np.diff(p_pa.values) > 0):
        u = u.sortby(p_dim)
        v = v.sortby(p_dim)
        omega = omega.sortby(p_dim)
        T = T.sortby(p_dim)
    p_pa = u[p_dim]

    latr = np.deg2rad(u[lat_dim])
    cosphi = xr.DataArray(np.cos(latr), dims=[lat_dim], coords={lat_dim: u[lat_dim]})
    f = xr.DataArray(2 * Omega * np.sin(latr), dims=[lat_dim], coords={lat_dim: u[lat_dim]})

    theta = T * (P0 / p_pa) ** (Rd / Cp)

    ubar = u.mean(lon_dim)
    thbar = theta.mean(lon_dim)

    up = u - ubar
    vp = v - v.mean(lon_dim)
    thp = theta - thbar

    vpthp_bar = (vp * thp).mean(lon_dim)

    # log-pressure height
    zstar = xr.DataArray(
        -H * np.log(p_pa / P0),
        dims=[p_dim],
        coords={p_dim: p_pa},
    )

    thbar_z = thbar.differentiate(p_dim) / zstar.differentiate(p_dim)
    thbar_z = thbar_z.where(np.abs(thbar_z) > 1e-10)

    # omega -> log-pressure vertical velocity
    zdot = -H * omega / p_pa
    zdotp = zdot - zdot.mean(lon_dim)
    zdotup_bar = (zdotp * up).mean(lon_dim)

    # density from zonal-mean T
    Tbar = T.mean(lon_dim)
    rho0 = p_pa / (Rd * Tbar)

    # absolute-vorticity-like bracket
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
Fz_era5 = compute_Fz_latmean(
    u_era5, v_era5, w_era5, t_era5,
    lat_bounds=(-5, 5),
    lon_dim="longitude",
    lat_dim="latitude",
    p_dim="level",
)
print(Fz_era5.level.values)
print("monthly...")
Fz_era5_monthly = Fz_era5.resample(time="MS").mean()
Fz_era5_monthly.name = "Fz"
Fz_era5_monthly.attrs["long_name"] = "Monthly mean vertical EP flux component"
Fz_era5_monthly.attrs["frequency"] = "monthly mean from daily ERA5 values"
print("loading into memory...")
Fz_era5_monthly = Fz_era5_monthly.load()

print("saving...")
outfile = f"/glade/derecho/scratch/kkoepnick/era5_daily/Fz_era5_{year}.nc"
Fz_era5_monthly.to_netcdf(outfile)

print("done :)")