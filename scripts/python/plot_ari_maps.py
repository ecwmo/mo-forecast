from datetime import timedelta
import pandas as pd
import xarray as xr
import salem
import xesmf as xe
from scipy.interpolate import interp1d

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt

from config import Config

""" ** INSERT RAINFALL TO ARI FUNCTION HERE ** """


def regrid(ds, _ds_out):
    return xe.Regridder(ds, _ds_out, "bilinear")(ds)


def mask(ds):
    conf = Config()
    _mask = salem.read_shapefile(
        conf.script_dir / "python/resources/shp/PHL_adm0/PHL_adm0.shp"
    )
    _ds_mask = ds.salem.roi(shape=_mask)

    return _ds_mask


def interpd(_newx, _vals, _idxs):
    return interp1d(_vals, _idxs, bounds_error=False, fill_value="extrapolate")(_newx)


def ARIinterp(_newx):
    conf = Config()
    _ph_ari_file = conf.script_dir / "python/resources/nc/PHIL_ARI.nc"
    _dat = xr.open_dataset(f"{_ph_ari_file}").rename({"precip": "rain"})

    _ari = [1.5, 2, 5, 10, 25, 30, 50, 100, 200, 500, 1000]
    _dat = _dat.assign_coords({"ari": _ari})
    _idx = xr.DataArray(_ari, dims=["ari"]).broadcast_like(_dat).to_dataset(name="rain")

    _ari_interpd = xr.apply_ufunc(
        interpd,
        _newx,
        _dat,
        _idx,
        input_core_dims=[["time"], ["ari"], ["ari"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
    )
    return _ari_interpd


def input_format(ds):
    conf = Config()

    _ds = ds.mean("ens").rename({"lon": "longitude", "lat": "latitude"})

    _trmm = (
        xr.open_dataset(conf.script_dir / "python/resources/nc/trmm_domain_regrid.nc")
        .rename({"precipitation": "rain", "lon": "longitude", "lat": "latitude"})
        .sel(longitude=slice(117.375, 126.375), latitude=slice(5.125, 18.875))
        .drop("time_bnds")
    )

    _ds_out = _trmm.drop(("time", "rain"))

    _wrf_inp = mask(regrid(_ds, _ds_out))
    _wrf_ari = ARIinterp(_wrf_inp)

    return _wrf_ari


""" ** COLORMAP ** """
plot_vars = {
    "rain": {
        "title": "Average Recurrence Interval (years)",
        "units": "years",
        "levels": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0],
        "colors": [
            "#ffffff",
            "#0064ff",
            "#01b4ff",
            "#32db80",
            "#9beb4a",
            "#ffeb00",
            "#ffb302",
            "#ff6400",
            "#eb1e00",
            "#af0000",
        ],
    },
}

""" ** START PLOTTING ** """
lon_formatter = LongitudeFormatter(zero_direction_label=True, degree_symbol="")
lat_formatter = LatitudeFormatter(degree_symbol="")

lon_labels = range(120, 130, 5)
lat_labels = range(5, 25, 5)
xlim = (116, 128)
ylim = (5, 20)
var_name = "rain"


def plot_ari(ds, out_dir):
    conf = Config()

    ds = input_format(ds[var_name].to_dataset())

    init_dt = pd.to_datetime(ds.time.values[0], utc=True).astimezone(conf.tz)
    init_dt_str = init_dt.strftime("%Y-%m-%d %H")

    var_info = plot_vars.get("rain")
    levels = var_info["levels"]
    colors = var_info["colors"]

    for t in range(conf.wrf_forecast_days):
        print(f"Day {t+1}...")

        fig = plt.figure(figsize=(8, 9), constrained_layout=True)
        ax = plt.axes(projection=conf.plot_proj)
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_xticks(lon_labels, crs=conf.plot_proj)
        ax.set_yticks(lat_labels, crs=conf.plot_proj)

        da = ds[var_name].isel(time=t)
        p = da.plot.pcolormesh(
            ax=ax,
            transform=conf.plot_proj,
            levels=levels,
            colors=colors,
            extend="both",
            add_labels=False,
            add_colorbar=False,
        )

        plt.colorbar(p, ticks=levels, shrink=0.5)
        p.colorbar.ax.set_title(f"[{var_info['units']}]", pad=20, fontsize=10)
        ax.coastlines()

        ax.set_extent((*xlim, *ylim))

        dt1 = pd.to_datetime(da.time.values, utc=True).astimezone(conf.tz)
        dt1_str = dt1.strftime("%Y-%m-%d %H")
        dt2 = dt1 + timedelta(days=1)
        dt2_str = dt2.strftime("%Y-%m-%d %H")
        plt_title = f"{var_info['title']}\nValid from {dt1_str} to {dt2_str} PHT"
        plt_annotation = f"WRF ensemble forecast initialized at {init_dt_str} PHT."

        fig.suptitle(plt_title, fontsize=14)
        ax.annotate(plt_annotation, xy=(5, -30), xycoords="axes points", fontsize=8)

        ax.annotate(
            "observatory.ph",
            xy=(10, 10),
            xycoords="axes points",
            fontsize=10,
            bbox=dict(boxstyle="square,pad=0.3", alpha=0.5),
            alpha=0.5,
        )

        tt = (t + 1) * 24
        out_file = out_dir / f"wrf-{tt}hr_ari_{init_dt.strftime('%Y-%m-%d_%H')}PHT.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close("all")
