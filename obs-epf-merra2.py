import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
import sys

# ---------------- settings ----------------
merra2_dir = Path("/glade/derecho/scratch/kkoepnick/merra2_daily")

year = int(sys.argv[1])
start = f"{year}-01-01"
end   = f"{year}-12-31"

# ---------------- helpers ----------------
def norm_time(ds):
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds

def wrap_lon(ds):
    if "longitude" in ds.coords:
        lon_name = "longitude"
    elif "lon" in ds.coords:
        lon_name = "lon"
    else:
        raise KeyError(f"No longitude coordinate found. Coords: {list(ds.coords)}")

    if float(ds[lon_name].max()) <= 180:
        ds = ds.assign_coords({lon_name: ds[lon_name] % 360}).sortby(lon_name)

    return ds

def get_merra2_files(year):
    files = sorted((merra2_dir / f"{year}").glob("MERRA2_*.inst3_3d_asm_Np.*.SUB.nc"))
    return [str(f) for f in files]

def preprocess_merra2(ds):
    ds = norm_time(ds)

    # Standardize names before subsetting.
    rename = {}
    if "lev" in ds.dims:
        rename["lev"] = "level"
    if "lat" in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims:
        rename["lon"] = "longitude"
    ds = ds.rename(rename)

    ds = wrap_lon(ds)
    ds = ds.sel(latitude=slice(-5, 5))

    return ds[["U", "V", "OMEGA", "T"]]

def load_merra2_all():
    files = get_merra2_files(year)

    print(f"loading MERRA-2 {year}: {len(files)} files")
    if not files:
        raise FileNotFoundError(f"No MERRA-2 files found in {merra2_dir / str(year)}")

    ds = xr.open_mfdataset(
        files,
        engine="h5netcdf",
        combine="by_coords",
        preprocess=preprocess_merra2,
        chunks={
            "time": 8,
            "level": -1,
            "latitude": -1,
            "longitude": 120,
        },
        parallel=False,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    ds = ds.sel(time=slice(start, end))
    ds = ds.sortby("level")
    # MERRA-2 lev is in hPa, same as ERA5/JRA-55 after renaming.
    print("levels:", ds.level.values)
    print("dims:", ds.dims)

    # Optional: daily mean first, to match your ERA5/JRA workflow.
    # If you prefer monthly from 3-hourly, delete these two lines.
    ds = ds.resample(time="1D").mean()

    return ds.chunk({"time": 365, "level": -1, "latitude": -1, "longitude": 120})

# ---------------- EP flux ----------------
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

    u, v, omega, T = xr.align(u, v, omega, T, join="inner")

    if not np.all(np.diff(u[p_dim].values) > 0):
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

# ---------------- run ----------------
print("loading variables...")
ds_merra2 = load_merra2_all()

u_merra2 = ds_merra2["U"]
v_merra2 = ds_merra2["V"]
w_merra2 = ds_merra2["OMEGA"]
t_merra2 = ds_merra2["T"]

print("calculating EPF_z...")
Fz_merra2 = compute_Fz_latmean(
    u_merra2, v_merra2, w_merra2, t_merra2,
    lat_bounds=(-5, 5),
    lon_dim="longitude",
    lat_dim="latitude",
    p_dim="level",
)

print("monthly...")
Fz_merra2_monthly = Fz_merra2.resample(time="MS").mean()
Fz_merra2_monthly.name = "Fz"
Fz_merra2_monthly.attrs["long_name"] = "Monthly mean vertical EP flux component"
Fz_merra2_monthly.attrs["frequency"] = "monthly mean from daily MERRA-2 values"

print("loading into memory...")
Fz_merra2_monthly = Fz_merra2_monthly.load()

print("saving...")
outfile = f"/glade/derecho/scratch/kkoepnick/merra2_daily/Fz_merra2_{year}.nc"
Fz_merra2_monthly.to_netcdf(outfile)

print("done :)")