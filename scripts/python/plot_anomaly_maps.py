import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
import calendar
import time
from datetime import datetime

import os
import sys
import glob
import getopt

import matplotlib as mp
import matplotlib.pyplot as plt

from config import Config
from helpers.anomaly_format import plot_proj, plot_vars, plot_format, plot_footer

conf = Config()
resources_dir = conf.script_dir / "python/resources/nc"
wrf_dir = conf.data_dir / "anomaly/nc"
inp_dir = os.getenv("WRF_SERVER_DIR")
out_dir = conf.data_dir / "anomaly"


def set_dates(months):
    global _init_date, _file_date

    _init_date = pd.to_datetime("today")  # - pd.DateOffset(months=1)
    _init_date = _init_date.strftime("%Y-%m-%d")
    _init_date = pd.date_range(end=_init_date, periods=months, freq="M").strftime(
        "%Y-%m"
    )

    ## compare existing files to get final dates
    _act_files = [
        os.path.basename(x) for x in glob.glob(f"{wrf_dir}/wrf_anomaly_????-??.nc")
    ]
    _act_files = pd.Index([_act_files[i][12:19] for i in range(len(_act_files))])
    _file_date = _init_date.difference(_act_files)

    return _file_date


def open_obs():  ## open observed climatological datasets
    gsmap = xr.open_dataset(f"{resources_dir}/gsmap_2001-2020_clim.nc")
    gsmap = gsmap.rename({"longitude": "lon", "latitude": "lat"})
    aphrodite = xr.open_dataset(f"{resources_dir}/APHRODITE_1971-2000_clim.nc")

    return xr.merge([gsmap, aphrodite])


def check_dates(date, ds):  ## check completeness of dates
    _day_end = calendar.monthrange(
        pd.to_datetime(date).year, pd.to_datetime(date).month
    )[1]

    _hours = 24
    _ds_hours = ds.time.to_dataset(name="times").groupby("time.day").count("time")
    _ds_hours = _ds_hours.where(_ds_hours["times"] < _hours, drop=True).day
    _ds_sub = ds.where(ds.time.dt.day.isin(_ds_hours.values), drop=True)
    _ds_sub = ds.drop_sel(time=_ds_sub["time"].values)
    _percent_mon = len(np.unique(_ds_sub["time.day"].values))
    _percent_mon = (_percent_mon / _day_end) * 100
    _percent_mon = round(_percent_mon, 2)

    print(f"Missing Days: {len(_ds_hours.values)}")
    ## add list of missing hours to text file
    _ref_dates = pd.date_range(start=f"{date}-01", periods=_hours * _day_end, freq="H")
    _act_dates = pd.Index(ds.time.values)
    _missing_dates = _ref_dates.difference(_act_dates)

    if not _missing_dates.empty:
        _missing_dates.to_series().to_csv(out_dir / f"txt/Missing_{date}.txt")
    else:
        print("No missing dates to save")

    if _percent_mon >= 80:
        _wrf_temp = _ds_sub[["temp"]].groupby("time.month").mean("time")
        _wrf_rain = _ds_sub[["rain"]].groupby("time.month").sum("time")

        _mask = xr.open_dataset(f"{resources_dir}/WRF_LANDMASK_PH.nc")
        _wrf_temp = _wrf_temp.assign(
            {
                "aave_temp": _wrf_temp["temp"]
                .where(_mask.LANDMASK == 1)
                .mean(("lat", "lon"))
            }
        )
        _wrf_rain = _wrf_rain.assign(
            {
                "aave_rain": _wrf_rain["rain"]
                .where(_mask.LANDMASK == 1)
                .mean(("lat", "lon"))
            }
        )

    return _wrf_temp.combine_first(_wrf_rain)


def _preprocess(
    ds,
):  ## function to select specific variables and time slice for open_mfdataset
    return ds.isel(time=slice(6))[["rain", "temp"]].mean("ens")


partial_func = partial(_preprocess)


def read_wrf_out(date):  ## open and process _file_dates
    _wrf_arowana = xr.open_mfdataset(
        f"{inp_dir}/output.arowana/nc/wrf_{date}-??_??.nc",
        concat_dim="time",
        combine="nested",
        preprocess=partial_func,
        chunks="auto",
        parallel=True,
    )

    _wrf_dugong = xr.open_mfdataset(
        f"{inp_dir}/output.dugong/nc/wrf_{date}-??_??.nc",
        concat_dim="time",
        combine="nested",
        preprocess=partial_func,
        chunks="auto",
        parallel=True,
    )
    _wrf_comb = _wrf_dugong.combine_first(_wrf_arowana)
    _wrf_month = check_dates(date, _wrf_comb)

    _obs = open_obs()

    _anom = _wrf_month[["temp", "rain"]] - _obs

    return {"actual": _wrf_month, "anomaly": _anom}


def plot_anom(save_nc=False, months=6):  ## plot anomalies per month
    start = time.time()

    set_dates(months)

    if not list(_file_date):
        print("▮▮▮ Files exist, nothing to plot.....")

    for date in _file_date:
        _anom = read_wrf_out(date)
        print(f"Plotting {date}")

        for var in ["rain", "temp"]:
            print(f"Plotting {var}....")

            fig, axes = plt.subplots(
                ncols=2, figsize=(10, 10), subplot_kw={"projection": plot_proj}
            )
            for ax, dat in zip(axes.flatten(), ["actual", "anomaly"]):
                var_info = plot_vars.get(f"{var}_{dat}")
                levels = var_info["levels"]
                colors = mp.colors.LinearSegmentedColormap.from_list(
                    "", var_info["colors"]
                )

                var_sub = _anom[dat][var].isel(month=0)
                p = var_sub.plot.contourf(
                    "lon",
                    "lat",
                    cmap=colors,
                    levels=levels,
                    extend="both",
                    add_labels=False,
                    add_colorbar=False,
                    transform=plot_proj,
                    ax=ax,
                )

                aave = _anom["actual"][f"aave_{var}"].isel(month=0).values.tolist()
                ax_ann = axes.flatten()[0]
                ax_ann.annotate(
                    f"AAVE: {np.round(aave, 2)}",
                    xy=(116.35, 19.2),
                    xycoords="data",
                    fontsize=10,
                    bbox=dict(boxstyle="square", fc="white", pad=0.3, alpha=0.5),
                )

                plot_format(ax)
                plot_footer(ax, var) if dat == "anomaly" else ""
                plt.colorbar(p, ax=ax, ticks=levels, shrink=0.35)
                p.colorbar.ax.set_title(f"[{var_info['units']}]", pad=20, fontsize=10)

                month_int = _anom["actual"].month.values[0]
                month_str = calendar.month_name[month_int]
                year_str = pd.to_datetime(date).year
                plt_title = f"{year_str} {month_str} {var_info['title']}"
                ax.set_title(plt_title, fontsize=10, y=1.025)

                fig.savefig(
                    out_dir / f"img/wrf_{var}_anomaly_{date}.png",
                    dpi=200,
                    facecolor="white",
                    bbox_inches="tight",
                )

                plt.close()

        if save_nc:
            print("Saving to netCDF")

            _wrf_nc = _anom["actual"].copy()
            _wrf_nc = _wrf_nc.assign(
                {
                    "anom_temp": _anom["anomaly"]["temp"],
                    "anom_rain": _anom["anomaly"]["rain"],
                }
            )
            _wrf_nc = _wrf_nc.rename({"month": "time"})
            _wrf_nc = _wrf_nc.assign_coords(
                {"time": np.atleast_1d(pd.to_datetime(date))}
            )
            _wrf_nc.load().to_netcdf(out_dir / f"nc/wrf_anomaly_{date}.nc")

        else:
            print("No data to save")

    end = time.time()
    elapsed = end - start

    print(
        f"▮▮▮ Elapsed time in real time : {datetime.strftime(datetime.utcfromtimestamp(elapsed), '%H:%M:%S')}"
    )


if __name__ == "__main__":
    save_nc = ""
    months = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hw:o:", ["save_nc=", "months="])

    except getopt.GetoptError:
        print("plot_anomaly_maps.py -w <save nc> -o <output dir>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("plot_anomaly_maps.py -w <save nc> -o <output dir>")
            sys.exit()

        elif opt in ("-w", "--save_nc"):
            save_nc = bool(arg)

        elif opt in ("-o", "--months"):
            months = int(arg)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "img"
    txt_dir = out_dir / "txt"
    nc_dir = out_dir / "nc"

    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    nc_dir.mkdir(parents=True, exist_ok=True)

    plot_anom(save_nc, months)
