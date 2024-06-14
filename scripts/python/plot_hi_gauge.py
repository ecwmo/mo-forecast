# title           :HI gauge plot
# description     :HI gauge plot
# author          :Danica Loqueloque <dloqueloque@observatory.ph>
# date            :20220328
# version         :2.0
# ==============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox

from config import Config


def dat_format(ds, site_lon, site_lat):
    conf = Config()
    _site_df = (
        ds.sel(lon=site_lon, lat=site_lat, method="nearest")
        .groupby("time")
        .mean("ens", skipna=True)
        .to_dataframe()
    )
    _site_df = (
        _site_df.tz_localize("utc")
        .tz_convert(conf.tz)
        .between_time("7:00", "18:00")
        .reset_index()
    )

    _data_reset = xr.Dataset.from_dataframe(_site_df.set_index("time"))
    _data_reset["time"] = _site_df["time"].dt.tz_localize(None).values

    _days = _data_reset.resample(time="1D").mean().time.dt.day.values.tolist()
    dat_formatted = []
    for i in _days:
        if ds.time.isel(time=0).time.dt.day == i:
            select = np.insert(
                _data_reset.hi.sel(time=_data_reset.time.dt.day == i), 0, 0
            ).fillna(0)
            select[1] = select[1] * 0
        else:
            select = _data_reset.hi.sel(time=_data_reset.time.dt.day == i).fillna(0)
        dat_formatted.append(select)

    return dat_formatted


def color(ds):
    _cols = []
    for i in ds.values:
        if 27.0 < i <= 32.0:
            _cols.append("#F0E786")
        elif 32.0 < i <= 41.0:
            _cols.append("#FF8C00")
        elif 41.0 < i <= 52.0:
            _cols.append("red")
        elif i > 52.0:
            _cols.append("purple")
        elif i < 27.0:
            _cols.append("#EEEEEE")

    if len(_cols) == 12:
        _cols.append("white")
    else:
        new_col = ["#EEEEEE"] * (12 - len(_cols)) + ["white"]
        _cols = _cols + new_col

    return _cols


def plot_gauge(ds, outdir):
    conf = Config()
    dat = pd.read_csv(conf.script_dir / "python/resources/csv/HI_cities.csv")

    for i in range(len(dat["NAME"])):
        station_name = dat["NAME"][i]
        site_lat = dat["Latitude"][i]
        site_lon = dat["Longitude"][i]

        _sub_ds = dat_format(ds, site_lon, site_lat)
        _fname = [
            _sub_ds[i].time.dt.strftime("%Y-%m-%d").isel(time=0).values.tolist()
            for i in range(len(_sub_ds[:3]))
        ]
        _day = [
            _sub_ds[i].time.dt.strftime("%a (%b %d)").isel(time=0).values.tolist()
            for i in np.arange(len(_sub_ds[:3]))
        ]
        _init_time = (
            pd.Series(ds["time"].values)
            .dt.tz_localize("utc")
            .dt.tz_convert(conf.tz)
            .dt.tz_localize(None)
        )

        fig, ax = plt.subplots(
            ncols=3,
            figsize=(16, 16),
            subplot_kw=dict(aspect="equal"),
            sharex=True,
            sharey=True,
        )
        plt.tight_layout(pad=0.0)
        fig.canvas.draw()
        fig.suptitle(
            f"{station_name} Heat Index \n ({_fname[0]} to { _fname[2]})",
            ha="center",
            y=0.73,
            fontsize=16,
        )

        axlist = ax.flatten()
        axlist[0].text(
            1.6,
            1.5,
            f"WRF ensemble forecast initialized at {_init_time.dt.strftime('%Y-%m-%d %H')[0]} PHT",
        )

        for i in range(3):
            axlist[i].set_title(_day[i])

        _dat = np.ones(12).tolist()
        _time = (
            pd.Series(pd.date_range("6:00", "18:00", freq="H"))
            .dt.strftime("%-I %p")
            .replace("12 PM", "12 NOON")
        ).tolist()
        _dat.append(12), _time.append("")

        _outer_colors = ["white"] * 13
        _inner_colors = [color(_sub_ds[i]) for i in range(len(_sub_ds[:3]))]

        _wedges, _text = axlist[0].pie(
            _dat,
            radius=0.93,
            colors=_outer_colors,
            counterclock=False,
            startangle=180,
            wedgeprops=dict(width=0.35, edgecolor="w", linewidth=0.6),
        )

        _d1 = axlist[0].pie(
            _dat,
            radius=0.93,
            colors=_inner_colors[0],
            counterclock=False,
            startangle=180,
            wedgeprops=dict(width=0.35, edgecolor="w", linewidth=0.6),
        )
        _d2 = axlist[1].pie(
            _dat,
            radius=0.93,
            colors=_inner_colors[1],
            counterclock=False,
            startangle=180,
            wedgeprops=dict(width=0.35, edgecolor="w", linewidth=0.6),
        )
        _d3 = axlist[2].pie(
            _dat,
            radius=0.93,
            colors=_inner_colors[2],
            counterclock=False,
            startangle=180,
            wedgeprops=dict(width=0.35, edgecolor="w", linewidth=0.6),
        )

        _nan1, _nan2, _nan3 = (
            [i for i, e in enumerate(_sub_ds[0].values.tolist()) if e == 0],
            [i for i, e in enumerate(_sub_ds[1].values.tolist()) if e == 0],
            [i for i, e in enumerate(_sub_ds[2].values.tolist()) if e == 0],
        )

        for i in _nan1:
            (
                _d1[0][i].set_hatch("..."),
                _d1[0][i].set_edgecolor("k"),
                _d1[0][i].set_alpha(0.4),
            )
        for i in _nan2:
            (
                _d2[0][i].set_hatch("..."),
                _d2[0][i].set_edgecolor("k"),
                _d2[0][i].set_alpha(0.4),
            )
        for i in _nan3:
            (
                _d3[0][i].set_hatch("..."),
                _d3[0][i].set_edgecolor("k"),
                _d3[0][i].set_alpha(0.4),
            )

        for i, p in enumerate(_wedges):
            y = np.sin(np.deg2rad(p.theta2))
            x = np.cos(np.deg2rad(p.theta2))

            for ax in axlist:
                if p.theta1 > 75:
                    ax.annotate(
                        _time[i],
                        xy=(x, y),
                        xytext=(0.5, 0.5),
                        textcoords="offset points",
                        ha="right",
                        va="center",
                        fontsize=9,
                    )

                elif round(p.theta1) == 75:
                    ax.annotate(
                        _time[i],
                        xy=(x, y),
                        xytext=(0.5, 0.5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                elif p.theta1 < 74:
                    ax.annotate(
                        _time[i],
                        xy=(x, y),
                        xytext=(0.5, 0.5),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=9,
                    )

        legends = [
            mpatches.Patch(color=cols, label=labels)
            for cols, labels in {
                "#F0E786": "Caution",
                "#FF8C00": "Extreme Caution",
                "red": "Danger",
                "purple": "Extreme Danger",
            }.items()
        ] + [
            mpatches.Patch(
                edgecolor="k", facecolor="w", alpha=0.4, hatch="...", label="No Data"
            )
        ]

        axlist[1].legend(
            handles=legends,
            loc=10,
            bbox_to_anchor=(0.25, 0.13, 0.5, 0.5),
            ncol=3,
            handleheight=1,
            handlelength=2,
        )

        out_file = outdir / f"{station_name}_WRF_HI_{_fname[0]}-{_fname[2]}.png"

        fig.savefig(
            out_file,
            dpi=300,
            facecolor="white",
            bbox_inches=Bbox([[0.0, 6.5], [16, 12]]),
        )
        plt.close()
