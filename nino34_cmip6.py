SST_CHUNKS = {"time": 120}

def compute_nino34_from_tos(tos_pattern):
    """
    Compute monthly Niño-3.4 anomalies (°C) from CMIP6 Omon/tos files.
    Works for both 1D and 2D lat/lon grids.
    Returns 1D DataArray (time) with name 'nino34'.
    """
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

    da = ds["tos"]  # CMIP6 SST name

    # Try to find lat / lon coordinate variables
    lat_candidates = [c for c in ds.coords if c.lower().startswith("lat")]
    lon_candidates = [c for c in ds.coords if c.lower().startswith("lon")]

    # Fallback for IPSL (nav_lat/nav_lon)
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

    # Handle lon range → 0–360 (works for 1D or 2D lon)
    lon360 = lon % 360

    # Decide whether we can use fast labeled slicing (lat/lon are *dimension* coords)
    can_slice = (lat_name in da.dims) and (lon_name in da.dims)

    if can_slice:
        # ---- DIMENSION COORDS: safe to .sel(slice) ----
        da2 = da.assign_coords({lon_name: lon360}).sortby(lon_name)

        # Ensure latitude monotonic increasing for slice
        if (da2[lat_name][0] > da2[lat_name][-1]).compute():
            da2 = da2.sortby(lat_name)

        sub = da2.sel(
            {lat_name: slice(-5, 5),
             lon_name: slice(190, 240)}
        )

        wlat = np.cos(np.deg2rad(sub[lat_name]))
        n34_mean = (
            sub.weighted(wlat)
               .mean(dim=lat_name)
               .mean(dim=lon_name)
        )

    else:
        # ---- AUX / CURVILINEAR COORDS: use mask ----
        mask = (
            (lat >= -5) & (lat <= 5) &
            (lon360 >= 190) & (lon360 <= 240)
        )

        # horizontal dims are whatever dims lat/lon live on (exclude time if present)
        horiz_dims = tuple(d for d in mask.dims if d != "time")

        weights = np.cos(np.deg2rad(lat)).where(mask).fillna(0)

        n34_mean = (
            da.where(mask)
              .weighted(weights)
              .mean(dim=horiz_dims)
        )
    # Monthly climatology and anomalies
    clim = n34_mean.groupby("time.month").mean("time")
    n34_anom = (n34_mean.groupby("time.month") - clim).rename("nino34")

    # Break weighted-wrapper, make it a plain DataArray
    n34_anom = n34_anom.copy(deep=True)
    ds.close()

    return n34_anom

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
    "UKESM1-0-LL":   "/glade/collections/cmip/CMIP6/CMIP/MOHC/UKESM1-0-LL/piControl/r1i1p1f2/Omon/tos/gn/v20190827/tos/*.nc",
}

N_YEARS  = 40
N_MONTHS = N_YEARS * 12

def build_nino34_stack_data(tos_patterns, n_months=N_MONTHS):
    """
    Returns dict keyed by model name:
      data[name] = {
        "y":     1D numpy array of Nino3.4 anomalies (last n_months),
        "sigma": std dev over that window,
        "x":     1D numpy array of years relative to end ([-N_YEARS, 0])
      }
    """
    data = {}

    for name, pat in tos_patterns.items():
        print(f"Computing Niño-3.4 for {name} ...")
        n34 = compute_nino34_from_tos(pat)

        # restrict
        n34_last = n34 #last_n_months(n34, n_months)

        # force numeric numpy array (compute dask if present)
        if hasattr(n34_last, "values"):
            y = np.asarray(n34_last.values, dtype=float)
        else:
            y = np.asarray(n34_last, dtype=float)

        sigma = np.nanstd(y)

        nmo = len(y)
        x = (np.arange(nmo) - (nmo - 1)) / 12.0  # ends at 0

        data[name] = {"y": y, "sigma": sigma, "x": x}

    return data

nino34_data = build_nino34_stack_data(tos_patterns, n_months=N_MONTHS)

with open("/glade/u/home/kkoepnick/qbo-enso/saved_data/nino34_cmip.pkl", "wb") as f:
    pickle.dump(nino34_data, f)

def remove_monthly_climatology(y):
    y = np.asarray(y, float)
    clim = np.full(12, np.nan)
    for m in range(12):
        clim[m] = np.nanmean(y[m::12])
    y_anom = y - clim[np.arange(len(y)) % 12]
    return y_anom, clim

def local_maxima_peaks(y, threshold):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    peaks = []
    for i in range(1, len(y) - 1):
        if not (ok[i-1] and ok[i] and ok[i+1]):
            continue
        if (y[i] > y[i-1]) and (y[i] > y[i+1]) and (y[i] >= threshold):
            peaks.append(i)
    return np.array(peaks, dtype=int)

def count_enso_peaks_per_month(nino34_data, threshold_sigma=1.0):
    """
    If threshold_sigma > 0: El Niño peaks = local maxima above +thr
    If threshold_sigma < 0: La Niña peaks = local minima below -thr (implemented via maxima of -y)
    Returns:
      counts_by_model: dict[name] -> (12,) counts
      counts_total:    (12,) total counts
    """
    counts_by_model = {}
    counts_total = np.zeros(12, dtype=int)

    for name, d in nino34_data.items():
        y = d["y"]
        sigma = d["sigma"]
        y_anom, _ = remove_monthly_climatology(y)

        thr = abs(threshold_sigma) * sigma

        if threshold_sigma > 0:
            peak_idx = local_maxima_peaks(y_anom, thr)
        else:
            peak_idx = local_maxima_peaks(-y_anom, thr)  # minima of y_anom

        months = peak_idx % 12  # assumes series starts in "Jan" bin; see note below

        counts = np.zeros(12, dtype=int)
        for m in months:
            counts[m] += 1

        counts_by_model[name] = counts
        counts_total += counts

    return counts_by_model, counts_total

el_nino_counts_by_model, _ = count_enso_peaks_per_month(nino34_data, threshold_sigma=+1.0)
la_nina_counts_by_model, _ = count_enso_peaks_per_month(nino34_data, threshold_sigma=-1.0)

with open("/glade/u/home/kkoepnick/qbo-enso/saved_data/el_nino_counts_by_model.pkl", "wb") as f:
    pickle.dump(el_nino_counts_by_model, f)

with open("/glade/u/home/kkoepnick/qbo-enso/saved_data/la_nina_counts_by_model.pkl", "wb") as f:
    pickle.dump(la_nina_counts_by_model, f)
