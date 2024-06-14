from cartopy import crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mp

import numpy as np

mp.rcParams["font.size"] = 9

plot_proj = ccrs.PlateCarree()


def plot_format(ax):
    lon_formatter = LongitudeFormatter(zero_direction_label=True, degree_symbol="")
    lat_formatter = LatitudeFormatter(degree_symbol="")

    lon_labels = range(120, 130, 5)
    lat_labels = range(5, 25, 5)
    xlim = (116, 128)
    ylim = (5, 20)

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(lon_labels, crs=plot_proj)
    ax.set_yticks(lat_labels, crs=plot_proj)
    ax.coastlines()
    ax.set_extent((*xlim, *ylim))

    ax.annotate(
        "observatory.ph",
        xy=(10, 10),
        xycoords="axes points",
        fontsize=7,
        bbox=dict(boxstyle="square,pad=0.3", alpha=0.5),
        alpha=0.65,
    )
    return ax


def plot_footer(ax, var):
    footer = "1971-2000 APHRODITE" if var == "temp" else "1998-2015 TRMM"
    ax.text(
        x=116,
        y=3.5,
        s=f"*baseline: {footer}",
        fontsize=7,
    )
    return ax


plot_vars = {
    "rain_actual": {
        "title": "Total Rainfall\n(WRF ensemble)",
        "units": "mm",
        "levels": np.arange(50, 500, 50),
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
    "rain_anomaly": {
        "title": "Total Rainfall Anomaly\n(WRF ensemble minus baseline*)",
        "units": "mm",
        "levels": np.arange(-150, 175, 25),
        "colors": [
            mp.colors.rgb2hex(mp.cm.get_cmap("BrBG")(i))
            for i in range(mp.cm.get_cmap("BrBG").N)
        ],
    },
    "temp_actual": {
        "title": "Average Temperature\n(WRF ensemble)",
        "units": "°C",
        "levels": np.arange(18, 34, 2),
        "colors": [
            mp.colors.rgb2hex(mp.cm.get_cmap("YlOrRd")(i))
            for i in range(mp.cm.get_cmap("YlOrRd").N)
        ],
    },
    "temp_anomaly": {
        "title": "Average Temperature\n(WRF ensemble minus baseline*)",
        "units": "°C",
        "levels": np.arange(-2.5, 3.0, 0.5),
        "colors": [
            mp.colors.rgb2hex(mp.cm.get_cmap("coolwarm")(i))
            for i in range(mp.cm.get_cmap("coolwarm").N)
        ],
    },
}
