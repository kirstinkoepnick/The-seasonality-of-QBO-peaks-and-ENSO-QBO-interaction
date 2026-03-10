import glob
import numpy as np
import xarray as xr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, lfilter, sosfilt, welch
import pickle 

### QBO seasonality CMIP6 ###


PLEV_HPA         = 50
LON_MIN, LON_MAX = 50, 280
LAT_BAND         = 10
CHUNKS           = {"time": 240}

def qbo_index_from_cmip(pattern, engine="netcdf4"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        decode_times=True,
        use_cftime=True,
        chunks=CHUNKS,
        parallel=False,       # <- key change
        engine=engine,     # <- optional but helps on Derecho
    )[["ua"]]                 # <- only data var; coords are still available

    if (ds.lon.min() < 0).item():
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")

    sub = ds.sel(
        lon=slice(LON_MIN, LON_MAX),
        lat=slice(-LAT_BAND, LAT_BAND)
    ).sortby("lat")

    ua50 = sub["ua"].sel(plev=PLEV_HPA*100, method="nearest")

    ua50_zm = ua50.mean("lon")
    wlat = np.cos(np.deg2rad(ua50_zm.lat))
    qbo_raw = ua50_zm.weighted(wlat).mean("lat")

    clim  = qbo_raw.groupby("time.month").mean("time")
    anom  = qbo_raw.groupby("time.month") - clim
    fit   = anom.polyfit(dim="time", deg=1)
    trend = xr.polyval(anom["time"], fit.polyfit_coefficients)
    qbo_detr = (anom - trend).astype("float32")

    qbo_detr = qbo_detr.load()
    ds.close()
    return qbo_detr

def bandpass_filter_1d_numpy(da, fs=1.0, low_per=36, high_per=12, order=6):
    """Bandpass a 1D DataArray along 'time' using filtfilt."""
    f_low = 1.0 / low_per
    f_high = 1.0 / high_per
    nyquist = 0.5 * fs
    low_cut = f_low / nyquist
    high_cut = f_high / nyquist

    b, a = butter(order, [low_cut, high_cut], btype='band')

    arr = da.values.astype("float32")         # 1D NumPy array
    filt = filtfilt(b, a, arr, axis=0, padlen=0)

    return xr.DataArray(
        filt,
        coords=da.coords,
        dims=da.dims,
        name=da.name
    )


def three_month_composite(ts1d):
    return ts1d.rolling(time=3, center=True, min_periods=3).mean().dropna("time")

def seasonal_counts_from_series(ts1d, peak_distance=12):
    comp = three_month_composite(ts1d.astype("float32"))
    y = comp.values
    east, _ = find_peaks(y,  distance=peak_distance)
    west, _ = find_peaks(-y, distance=peak_distance)
    east_months = comp.time.dt.month.values[east]
    west_months = comp.time.dt.month.values[west]
    east_counts = np.bincount(east_months - 1, minlength=12)
    west_counts = np.bincount(west_months - 1, minlength=12)
    return east_counts, west_counts

# patterns per model
model_patterns = {
    #### CMIP6 Models: ####
    "AWI-CM-1-1-MR":    "/glade/collections/cmip/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/piControl/r1i1p1f1/Amon/ua/gn/v20181218/ua/*.nc",
    "BCC-CSM2-MR":      "/glade/collections/cmip/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Amon/ua/gn/v20181016/ua/*.nc",
    "CESM2-WACCM":      "/glade/collections/cmip/CMIP6/CMIP/NCAR/CESM2-WACCM/piControl/r1i1p1f1/Amon/ua/gn/v20190320/*.nc",
    "CNRM-CM6-1":       "/glade/collections/cmip/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/piControl/r1i1p1f2/Amon/ua/gr/v20180814/ua/*.nc",
    "CNRM-ESM2-1":      "/glade/collections/cmip/CMIP6/CMIP/CNRM-CERFACS/CNRM-ESM2-1/piControl/r1i1p1f2/Amon/ua/gr/v20181115/ua/*.nc", 
    "E3SM-1-0":         "/glade/collections/cmip/CMIP6/CMIP/E3SM-Project/E3SM-1-0/piControl/r1i1p1f1/Amon/ua/gr/v20190723/ua/*.nc",
    "EC-Earth3":        "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/EC-Earth3/ua/Amon/ua*.nc",
    # corrupted!! "EC-Earth3-Veg":    "/glade/collections/cmip/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3-Veg/piControl/r1i1p1f1/Amon/ua/gr/v20190619/ua/*.nc", corrupted!!!
    "GFDL-ESM4":        "/glade/collections/cmip/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/piControl/r1i1p1f1/Amon/ua/gr1/v20180701/ua/*.nc",
    "HadGEM3-GC31-LL":  "/glade/collections/cmip/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/piControl/r1i1p1f1/Amon/ua/gn/v20190628/ua/*.nc",
    "IPSL-CM6A-LR":     "/glade/collections/cmip/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Amon/ua/gr/v20181123/ua/*.nc",
    "MIROC6":           "/glade/collections/cmip/CMIP6/CMIP/MIROC/MIROC6/piControl/r1i1p1f1/Amon/ua/gn/v20190311/ua/*.nc",
    "MPI-ESM1-2-HR":    "/glade/campaign/univ/uhar0025/kkoepnick/cmip6/MPI-ESM1-2-HR/ua/Amon/ua*.nc", 
    "MRI-ESM2-0":       "/glade/collections/cmip/CMIP6/CMIP/MRI/MRI-ESM2-0/piControl/r1i1p1f1/Amon/ua/gn/v20190308/ua/*.nc",
    "UKESM1-0-LL":      "/glade/collections/cmip/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl/r1i1p1f2/Amon/ua/gn/v20190410/ua/*.nc"
}

results = {}
qbo_bp_results = {}
for name, pattern in model_patterns.items():
    print(f"Processing {name} ...")
    if name == "EC-Earth3-Veg":
        qbo = qbo_index_from_cmip(pattern, engine="h5netcdf")               # 1D, already loaded
    else:
        qbo = qbo_index_from_cmip(pattern)
    qbo_bp = bandpass_filter_1d_numpy(
        qbo, low_per=36, high_per=12, order=3
    )
    
    east, west = seasonal_counts_from_series(qbo_bp)
    qbo_bp_results[name] = qbo_bp
    results[name] = (east, west)

import pickle

with open("/glade/u/home/kkoepnick/qbo-enso/saved_data/qbo_bp_results.pkl", "wb") as f:
    pickle.dump(qbo_bp_results, f)

with open("/glade/u/home/kkoepnick/qbo-enso/saved_data/qbo_seasonal_peaks_CMIP.pkl", "wb") as f:
    pickle.dump(results, f)

def fit_annual_sine(months, y):
    """Fit y(m) with a single annual harmonic and return amplitude & phase."""
    omega = 2 * np.pi / 12.0  # annual cycle

    # Design matrix: sin, cos, constant
    X = np.column_stack([
        np.sin(omega * months),
        np.cos(omega * months),
        np.ones_like(months)
    ])

    (a, b, c), *_ = np.linalg.lstsq(X, y, rcond=None)

    # Convert to amplitude/phase form
    A = np.hypot(a, b)
    phi = np.arctan2(b, a)
    m_peak = (0.5 * np.pi - phi) / omega  # month where sine hits max

    y_fit = A * np.sin(omega * months + phi) + c

    return A, phi, c, m_peak, y_fit

# ---- Fit all models and store the results ----
months = np.arange(1, 13)

fit_results = {}

for name, (east_counts, west_counts) in results.items():

    A_e, phi_e, c_e, mpeak_e, yfit_e = fit_annual_sine(months, east_counts)
    A_w, phi_w, c_w, mpeak_w, yfit_w = fit_annual_sine(months, west_counts)

    fit_results[name] = {
        "east": {
            "A": A_e, "phi": phi_e, "offset": c_e,
            "peak_month": (mpeak_e - 1) % 12 + 1,
            "fit_curve": yfit_e,
        },
        "west": {
            "A": A_w, "phi": phi_w, "offset": c_w,
            "peak_month": (mpeak_w - 1) % 12 + 1,
            "fit_curve": yfit_w,
        },
    }

# ---- Fit all models and store the results ----
months = np.arange(1, 13)

fit_results_prob = {}

for name, (east_counts, west_counts) in results.items():
    east_prob = east_counts/nyears[name] #/np.sum(east_counts)
    west_prob = west_counts/nyears[name] #/np.sum(west_counts)
    A_e, phi_e, c_e, mpeak_e, yfit_e = fit_annual_sine(months, east_prob)
    A_w, phi_w, c_w, mpeak_w, yfit_w = fit_annual_sine(months, west_prob)

    fit_results_prob[name] = {
        "east": {
            "A": A_e, "phi": phi_e, "offset": c_e,
            "peak_month": (mpeak_e - 1) % 12 + 1,
            "fit_curve": yfit_e,
        },
        "west": {
            "A": A_w, "phi": phi_w, "offset": c_w,
            "peak_month": (mpeak_w - 1) % 12 + 1,
            "fit_curve": yfit_w,
        },
    }

fit_results_prob

with open("/glade/u/home/kkoepnick/qbo-enso/saved_data/seasonal_fit.pkl", "wb") as f:
    pickle.dump(fit_results, f)

with open("/glade/u/home/kkoepnick/qbo-enso/saved_data/seasonal_fit_prob.pkl", "wb") as f:
    pickle.dump(fit_results_prob, f)
