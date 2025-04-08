from datetime import timedelta
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from config import Config

# MOIP
# station_name = "Manila Observatory"
# site_lon = 121.077
# site_lat = 14.64


def plot_timeseries(ds, out_dir):
    conf = Config()
    init_dt = pd.to_datetime(ds.time.values[0], utc=True).astimezone(conf.tz)

    cities = pd.read_csv(conf.script_dir / "python/resources/csv/cities.csv")
    for idx in cities.index:
        station_name = cities["name"][idx]
        site_lat = cities["lat"][idx]
        site_lon = cities["lon"][idx]
        print(f"Plotting timeseries for {station_name}: {site_lon} {site_lat}")

        site_df = ds.sel(lon=site_lon, lat=site_lat, method="nearest").to_dataframe()
        site_df = site_df.groupby("time").mean()
        site_df.index = site_df.index.tz_localize("utc").tz_convert(conf.tz)

        x_range = range(0, site_df.shape[0])
        t_range = pd.date_range(init_dt, periods=len(x_range), freq="H")

        x_minor_step = 6
        t_major_ticks = t_range[t_range.hour == 0]
        t_minor_ticks = t_range[(t_range.hour % x_minor_step) == 0]
        x_major_ticks = [t_range.get_loc(t) for t in t_major_ticks]
        x_minor_ticks = [t_range.get_loc(t) for t in t_minor_ticks]

        fig, axs = plt.subplots(
            ncols=1, nrows=5, figsize=(25, 25), constrained_layout=True
        )

        # region plot rainfall
        site_df.plot.bar(
            y="rain", edgecolor="blue", facecolor="blue", ax=axs[0], label="_", zorder=5
        )
        _ax = axs[0].twiny()

        _t_major_ticks = t_major_ticks - timedelta(hours=12)
        _ax.set_xticks([x - 12 for x in x_major_ticks])
        _ax.set_xticklabels(_t_major_ticks.strftime("%A %b %-d"))
        _ax.tick_params(axis="x", labelsize=22, top=False)
        _ax.set_xlabel(None)
        _ax.set_xlim([-2, 121])

        axs[0].set_xticks(x_major_ticks)
        axs[0].set_xticklabels(t_major_ticks.strftime("%a %H:00"))
        axs[0].set_xticks(x_minor_ticks, minor=True)
        axs[0].set_xticklabels(
            t_minor_ticks.strftime("%a %H:00"),
            minor=True,
        )
        axs[0].tick_params(axis="x", which="both", labelsize=22, labelrotation=30)
        axs[0].set_xlabel(None)
        axs[0].set_xlim([-2, 121])

        y_major_step = 10
        y_minor_step = 5
        axs[0].set_yticks(range(0, 45, y_major_step))
        axs[0].set_yticks(range(0, 45, y_minor_step), minor=True)
        axs[0].tick_params(axis="y", labelsize=24)
        axs[0].set_ylabel("Rainfall (mm/hr)", fontsize=24, color="blue")
        axs[0].set_ylim([-0.1, 40])

        axs[0].grid(which="major", axis="x", c="gray", ls="dotted")
        axs[0].grid(which="minor", axis="y", c="gray", ls="dotted")

        axs[0].fill_between(
            x_range,
            7.5,
            15,
            color="yellow",
            alpha=0.5,
            label="Heavy (7.5-15 mm/hr)",
        )
        axs[0].fill_between(
            x_range,
            15,
            30,
            color="orange",
            alpha=0.5,
            label="Intense (15-30 mm/hr)",
        )
        axs[0].fill_between(
            x_range,
            30,
            50,
            color="red",
            alpha=0.5,
            label="Torrential (> 30 mm/hr)",
        )

        axs[0].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.8),
            fontsize=20,
            ncol=3,
            title_fontsize="20",
            title="Colors in the rainfall time series are based on DOST-PAGASA's color-coded rainfall advisories and classification:",
        )
        axs[0].set_title(
            f"Ensemble Forecast ({station_name})\nInitialized at {init_dt.strftime('%Y-%m-%d %H')}:00 PHT",
            pad=38,
            fontsize=30,
        )
        # endregion plot rainfall

        # region plot temperature and heat index
        site_df.plot(
            y="hi",
            color="purple",
            marker=".",
            lw=3,
            markersize=12,
            label="Heat Index (°C)",
            ax=axs[1],
        )
        site_df.plot(
            y="temp",
            color="red",
            marker=".",
            lw=3,
            markersize=12,
            label="Temperature (°C)",
            ax=axs[1],
        )

        axs[1].set_xticks(t_major_ticks)
        axs[1].set_xticklabels([])
        axs[1].set_xticks(t_minor_ticks, minor=True)
        axs[1].set_xticklabels([], minor=True)
        axs[1].tick_params(axis="x", labelsize=18)
        axs[1].set_xlabel(None)
        axs[1].set_xlim([site_df.index.min() - timedelta(hours=2), site_df.index.max()])

        axs[1].tick_params(axis="y", labelsize=24)
        axs[1].set_ylabel("Temp/HI (°C)", fontsize=24, color="red")
        y_min = site_df[["temp", "hi"]].min().min()
        y_min = np.floor(y_min / 5.0) * 5
        y_max = site_df[["temp", "hi"]].max().max()
        y_max = np.ceil(y_max / 5.0) * 5
        axs[1].set_ylim([y_min, y_max])
        axs[1].set_yticks(range(int(y_min), int(y_max) + 1, 5))

        axs[1].grid(which="major", c="gray", ls="dotted")

        axs[1].legend(
            loc="lower center", bbox_to_anchor=(0.415, -0.22), fontsize=20, ncol=2
        )
        # endregion plot temperature and heat index

        # region plot relative humidity
        _ax = axs[1].twinx()
        site_df.plot(
            y="rh",
            color="green",
            marker=".",
            lw=3,
            markersize=12,
            label="Relative Humidity",
            ax=_ax,
        )

        _ax.set_xticks(t_major_ticks)
        _ax.set_xticklabels([])
        _ax.set_xticks(t_minor_ticks, minor=True)
        _ax.set_xticklabels([], minor=True)
        _ax.tick_params(axis="x", labelsize=18)
        _ax.set_xlabel(None)
        _ax.set_xlim([site_df.index.min() - timedelta(hours=2), site_df.index.max()])

        _ax.tick_params(axis="y", labelsize=24)
        _ax.set_ylabel("Relative Humidity (%)", fontsize=24, color="green")
        _ax.set_ylim([20, 100])

        _ax.legend(
            loc="lower center", bbox_to_anchor=(0.65, -0.22), fontsize=20, ncol=2
        )
        # endregion plot relative humidity

        # region plot wind speed and wind barbs
        wspd = (site_df.u_850hPa**2 + site_df.v_850hPa**2) ** 0.5
        wspd.plot(
            color="black",
            marker=".",
            lw=3,
            markersize=12,
            ax=axs[2],
        )
        y_range = [20] * len(x_range)
        axs[2].quiver(
            t_range,
            y_range,
            site_df.u_850hPa / wspd,
            site_df.v_850hPa / wspd,
            width=0.001,
            color="black",
            units="width",
            scale=50,
            headwidth=5,
        )

        axs[2].set_xticks(t_major_ticks)
        axs[2].set_xticklabels(t_major_ticks.strftime("%a %H:00"))
        axs[2].set_xticks(t_minor_ticks, minor=True)
        axs[2].set_xticklabels(
            t_minor_ticks.strftime("%a %H:00"),
            minor=True,
        )
        axs[2].tick_params(axis="x", which="both", labelsize=22, labelrotation=30)
        axs[2].set_xlabel("Day and Time (PHT)", fontsize=24)
        axs[2].set_xlim([site_df.index.min() - timedelta(hours=2), site_df.index.max()])

        axs[2].tick_params(axis="y", labelsize=24)
        axs[2].set_ylabel("Wind Speed (m/s)", fontsize=24, color="black")
        axs[2].set_ylim([0, 40])

        axs[2].grid(which="major", axis="x", c="gray", ls="dotted")
        # endregion plot wind speed and wind barbs

        ############### clean energy forecast ##################
        ########################################################

        # region plot WPD
        site_df.plot.bar(
            y="wpd",
            edgecolor="red",
            facecolor="red",
            ax=axs[3],
            label="Wind Power Potential",
        )

        _ax = axs[3].twiny()
        _t_major_ticks = t_major_ticks - timedelta(hours=12)
        _ax.set_xticks([x - 12 for x in x_major_ticks])
        _ax.set_xticklabels(_t_major_ticks.strftime("%A %b %-d"))
        _ax.tick_params(axis="x", labelsize=22, top=False)
        _ax.set_xlabel(None)
        _ax.set_xlim([-2, 121])

        axs[3].set_xticks(x_major_ticks)
        axs[3].set_xticklabels([])
        axs[3].set_xticks(x_minor_ticks, minor=True)
        axs[3].set_xticklabels(
            [],
            minor=True,
        )
        axs[3].set_xlabel(None)
        axs[3].set_xlim([-2, 121])

        axs[3].set_yticks(range(0, 16, 2))
        axs[3].tick_params(axis="y", labelsize=24)
        axs[3].set_ylabel("Wind PP (MW)", fontsize=24, color="black")
        y_max = site_df.wpd.max()
        if y_max <= 4:
            y_max = 4
        y_max = round(float(y_max) / 2) * 2
        axs[3].set_ylim([0, y_max])

        axs[3].grid(which="major", c="gray", ls="dotted")

        axs[3].legend(loc="upper right", fontsize=24)
        axs[3].set_title(
            f"Clean Power Ensemble Forecast ({station_name})\nInitialized at {init_dt.strftime('%Y-%m-%d %H')}:00 PHT",
            pad=38,
            fontsize=30,
        )
        # endregion plot WPD

        # region plot PPV
        site_df.plot.bar(
            y="ppv",
            edgecolor="goldenrod",
            facecolor="goldenrod",
            ax=axs[4],
            label="Solar Power Potential",
        )

        axs[4].set_xticks(x_major_ticks)
        axs[4].set_xticklabels(t_major_ticks.strftime("%a %H:00"))
        axs[4].set_xticks(x_minor_ticks, minor=True)
        axs[4].set_xticklabels(
            t_minor_ticks.strftime("%a %H:00"),
            minor=True,
        )
        axs[4].tick_params(axis="x", which="both", labelsize=22, labelrotation=30)
        axs[4].set_xlabel("Day and Time (PHT)", fontsize=24)
        axs[4].set_xlim([-2, 121])

        axs[4].set_yticks(np.linspace(0, 2, 5))
        axs[4].tick_params(axis="y", labelsize=24)
        axs[4].set_ylabel("Solar PP (MW)", fontsize=24, color="black")
        axs[4].set_ylim([0, 2])

        axs[4].grid(which="major", c="gray", ls="dotted")

        axs[4].legend(loc="upper right", fontsize=24)
        # endregion plot PPV

        out_file = (
            out_dir
            / f"wrf-ts_{station_name.replace(' ', '')}_{init_dt.strftime('%Y-%m-%d_%H')}PHT.png"
        )
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close("all")
