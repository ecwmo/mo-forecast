from salem import open_xr_dataset
import seaborn as sns

from config import Config


conf = Config()

domain_land_mask = open_xr_dataset(conf.script_dir / "python/resources/nc/mask.nc")
ph_land_mask = open_xr_dataset(conf.script_dir / "python/resources/nc/ph_mask.nc")
trmm = open_xr_dataset(conf.script_dir / "python/resources/nc/trmm_1998-2015_clim.nc")[
    "precipitation"
]

rain_color = [
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
]

rain_levs_24hr = [5, 10, 20, 30, 50, 100, 150, 200, 250]
rain_levs_3hr = [5, 10, 20, 30, 50, 75, 100, 125, 150]

plot_vars = {
    "temp": {
        "title": "Air (2m) and Sea Surface Temperature [°C]",
        "units": "°C",
        "levels": range(20, 42, 2),
        "colors": sns.blend_palette(
            ["#f0e68c", "#ff4500", "#9932cc", "#000000"], n_colors=12
        ),
    },
    "hi": {
        "title": "Heat Index [°C]",
        "units": "°C",
        "levels": [27, 32, 41, 52],
        "colors": sns.blend_palette(
            ["#ffffff", "#f0e68c", "#ff8c00", "#b22222", "#9932cc"], n_colors=5
        ),
    },
    "hix": {
        "title": "Maximum Heat Index [°C]",
        "units": "°C",
        "levels": [27, 32, 41, 52],
        "colors": sns.blend_palette(
            ["#ffffff", "#f0e68c", "#ff8c00", "#b22222", "#9932cc"], n_colors=5
        ),
    },
    "rh": {
        "title": "Relative Humidity (2m)[%]",
        "units": "%",
        "levels": range(30, 110, 10),
        "colors": sns.blend_palette(
            [
                "#d7191c",
                "#fdae61",
                "#ffffbf",
                "#abdda4",
                "#2b83ba",
            ],
            n_colors=9,
        ),
    },
    "wind": {
        "title": "Winds (850mb)[m/s]",
        "units": "m/s",
        "levels": range(10, 70, 10),
        "colors": [(1, 1, 1)]
        + sns.blend_palette(
            [
                "#2b83ba",
                "#abdda4",
                "#ffffbf",
                "#fdae61",
                "#d7191c",
            ],
            n_colors=6,
        ),
    },
    "wpd": {
        "title": "-Hr Total Wind Power Potential [MW/hectare]",
        "units": "MW/\nhectare",
        "levels": [2.1, 3.2, 4.2, 5.2, 6.2, 8.3],
        "colors": [
            "#ffffff",
            "#ffb302",
            "#ffeb00",
            "#9beb4a",
            "#32db80",
            "#01b4ff",
            "#0064ff",
            "#000096",
        ],
    },
}

plot_vars_24hr = {
    "rain": {
        "title": "24-Hr Total Rainfall [mm/24hrs]",
        "units": "mm/\n24hrs",
        "levels": rain_levs_24hr,
        "colors": rain_color,
        "ens_mem": True,
    },
    "rainx": {
        "title": "Areas with Potential Extreme Rainfall",
        "units": "mm/\n24hrs",
        "levels": rain_levs_24hr,
        "colors": rain_color,
    },
    "rainx_clim": {
        "title": "Areas with Potential Extreme Rainfall",
        "units": "mm/\n24hrs",
        "levels": rain_levs_24hr,
        "colors": rain_color,
    },
    "ppv": {
        "title": "-Hr Total Solar Power Potential [MW/hectare]",
        "units": "MW/\nhectare",
        "levels": [1, 2.4, 3.8, 5.2, 6.6, 8],
        "colors": sns.blend_palette(
            [
                "#66cdaa",
                "#f0e68c",
                "#ff8c00",
                "#b22222",
            ],
            n_colors=7,
        ),
    },
}

plot_vars_3hr = {
    "rain": {
        "title": "3-Hr Total Rainfall [mm/3hrs]",
        "units": "mm/\n3hrs",
        "levels": rain_levs_3hr,
        "colors": rain_color,
        "ens_mem": True,
    },
    "rainx": {
        "title": "Areas with Potential Extreme Rainfall",
        "units": "mm/\n3hrs",
        "levels": rain_levs_3hr,
        "colors": rain_color,
    },
    "ppv": {
        "title": "-Hr Total Solar Power Potential [MW/hectare]",
        "units": "MW/\nhectare",
        "levels": [1, 1.6, 2.2, 2.8, 3.4, 4],
        "colors": sns.blend_palette(
            [
                "#66cdaa",
                "#f0e68c",
                "#ff8c00",
                "#b22222",
            ],
            n_colors=7,
        ),
    },
}

plot_vars_web = {
    "wpd": {
        "levels": [2.1, 3.2, 4.2, 5.2, 6.2, 8.3],
        "colors": [
            "#ffffff",
            "#ffb302",
            "#ffeb00",
            "#9beb4a",
            "#32db80",
            "#01b4ff",
            "#0064ff",
            "#000096",
        ],
    },
    "temp": {
        "levels": range(20, 42, 2),
        "colors": sns.blend_palette(
            ["#f0e68c", "#ff4500", "#9932cc", "#000000"], n_colors=12
        ),
    },
    "rainchance": {
        "levels": [i / 3 for i in range(conf.num_ens_member + 1)],
        "colors": ["#ffffff"]
        + sns.blend_palette(
            [
                "#b0e0e6",
                "#1e90ff",
                "#000080",
            ],
            n_colors=conf.num_ens_member,
        ),
    },
    "wind": {
        "levels": range(10, 70, 10),
        "colors": sns.color_palette(
            "RdPu",
            n_colors=7,
        ),
    },
    "ppv": {
        "levels": [1, 2.4, 3.8, 5.2, 6.6, 8],
        "colors": sns.blend_palette(
            [
                "#66cdaa",
                "#f0e68c",
                "#ff8c00",
                "#b22222",
            ],
            n_colors=7,
        ),
    },
}
