from datetime import timedelta
import pandas as pd

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt

from const import (
    plot_vars,
    plot_vars_3hr,
    plot_vars_24hr,
    domain_land_mask as land_mask,
)
from config import Config
from helpers.ari import regrid, open_ref, mask, run_interp
from helpers.model_agreement import model_agreement

lon_formatter = LongitudeFormatter(zero_direction_label=True, degree_symbol="")
lat_formatter = LatitudeFormatter(degree_symbol="")

lon_labels = range(120, 130, 5)
lat_labels = range(5, 25, 5)
xlim = (116, 128)
ylim = (5, 20)


def plot_maps(ds, out_dir):
    conf = Config()
    init_dt = pd.to_datetime(ds.time.values[0], utc=True).astimezone(conf.tz)
    init_dt_str = init_dt.strftime("%Y-%m-%d %H")
    tsteps = len(ds.time)
    hr_interval = int(24 * conf.wrf_forecast_days / tsteps)

    plot_var_items = plot_vars
    if hr_interval == 3:
        plot_var_items.update(plot_vars_3hr)
    else:
        plot_var_items.update(plot_vars_24hr)
    for var_name, var_info in plot_var_items.items():
        print(f"Plotting {var_name}...")

        levels = var_info["levels"]
        colors = var_info["colors"]
        interval_str = str(hr_interval)
        units = f"[{var_info['units']}]"
        if var_name in ["wpd", "ppv"]:
            title = interval_str + var_info["title"]
        else:
            title = var_info["title"]

        plot_ens_mem = var_info.get("ens_mem", False)
        for t in range(tsteps):
            print(f"Timestep {t+1}...")

            if var_name in ["rain", "rh", "wpd", "ppv"]:
                das = {"ens": ds[var_name].isel(time=t).mean("ens")}
                if plot_ens_mem:
                    for ens_idx in ds[var_name].ens.values:
                        das[f"run{int(ens_idx)+1}"] = ds[var_name].isel(
                            time=t, ens=ens_idx
                        )
            elif var_name in ["rainx", "rainx_clim"]:
                das = {"ens": ds["rain"].isel(time=t).mean("ens")}
                das_regrid = mask(regrid(ds["rain"].mean("ens"), open_ref()))
            elif var_name == "temp":
                _da = (
                    ds["tsk"]
                    .isel(time=t)
                    .mean("ens")
                    .salem.roi(roi=land_mask.mask.isnull())
                )
                _das = {"ens": _da}
                da = ds[var_name].isel(time=t).mean("ens").salem.roi(roi=land_mask.mask)
                da.values = _da.fillna(0).values + da.fillna(0).values
                das = {"ens": da}
            elif var_name in ["hi", "hix"]:
                _da = ds[var_name].isel(time=t).mean("ens")
                das = {"ens": _da.salem.roi(roi=land_mask.mask, crs=conf.plot_proj)}
            elif var_name == "wind":
                _u = ds["u_850hPa"].isel(time=t).mean("ens")
                _v = ds["v_850hPa"].isel(time=t).mean("ens")
                u = _u[::20, ::20]
                v = _v[::20, ::20]
                da = _u.copy()
                da.values = (_u.values**2 + _v.values**2) ** 0.5
                das = {"ens": da}
            else:
                continue

            for da_name, da in das.items():
                dt1 = pd.to_datetime(da.time.values, utc=True).astimezone(conf.tz)
                dt1_str = dt1.strftime("%Y-%m-%d %H")
                dt2 = dt1 + timedelta(hours=hr_interval)
                dt2_str = dt2.strftime("%Y-%m-%d %H")
                plt_title = f"{title}\nValid from {dt1_str} to {dt2_str} PHT"
                plt_annotation = (
                    f"WRF ensemble forecast initialized at {init_dt_str} PHT."
                )

                fig_scale = 0.75
                fig = plt.figure(
                    figsize=(8 * fig_scale, 9 * fig_scale), constrained_layout=True
                )
                ax = plt.axes(projection=conf.plot_proj)
                ax.xaxis.set_major_formatter(lon_formatter)
                ax.yaxis.set_major_formatter(lat_formatter)
                ax.set_xticks(lon_labels, crs=conf.plot_proj)
                ax.set_yticks(lat_labels, crs=conf.plot_proj)
                ax.coastlines()
                ax.set_extent((*xlim, *ylim))

                if var_name == "rainx":
                    filt_ari = run_interp(das_regrid).isel(time=t)
                    filt_ari = regrid(
                        filt_ari, ds[["lat", "lon"]], method="conservative"
                    )
                    da = da.where(filt_ari["rain"] >= 5)

                elif var_name == "rainx_clim":
                    filt_month = model_agreement(ds["rain"].isel(time=t))
                    da = da.where(filt_month >= 2)

                elif var_name == "temp":
                    p = _das[da_name].plot.contour(
                        ax=ax,
                        transform=conf.plot_proj,
                        levels=range(28, 32),
                        colors="#ffffff",
                        add_labels=False,
                        linewidths=0.5,
                    )
                    ax.clabel(p, p.levels, inline=True, fontsize=6 * fig_scale)
                elif var_name == "wind":
                    plt.streamplot(
                        u.lon.values,
                        u.lat.values,
                        u.values,
                        v.values,
                        density=2,
                        color="k",
                        linewidth=0.5,
                        transform=conf.plot_proj,
                    )

                    p = da.plot.contourf(
                        ax=ax,
                        transform=conf.plot_proj,
                        levels=levels,
                        colors=colors,
                        add_labels=False,
                        extend="both",
                        cbar_kwargs=dict(shrink=0.5),
                    )
                    plt.gca().set_facecolor("grey")
                elif var_name == "ppv":
                    da = da.salem.roi(roi=land_mask.mask)

                fig.suptitle(plt_title, fontsize=14 * fig_scale)

                if var_name in ["wpd", "ppv"]:
                    text_color = "blue"
                    if var_name == "ppv":
                        text_color = "darkorange"

                    tot_wpd = da.salem.roi(roi=land_mask.mask).sum().values / 1000
                    tot_wpd = int(tot_wpd.round())
                    ax.set_title(
                        f"Total$^{{*}}$: {tot_wpd} GW",
                        fontsize=24 * fig_scale,
                        color=text_color,
                    )

                if var_name != "wind":
                    p = da.plot.contourf(
                        ax=ax,
                        transform=conf.plot_proj,
                        levels=levels,
                        colors=colors,
                        add_labels=False,
                        extend="both",
                        cbar_kwargs=dict(shrink=0.5),
                    )
                if var_name == "rainx":  # from previous extreme < 2 to ari >=5
                    da.plot.contourf(
                        ax=ax,
                        transform=conf.plot_proj,
                        levels=levels,
                        colors=colors,  # previously white
                        alpha=1,
                        extend="both",
                        add_labels=False,
                        add_colorbar=False,
                    )
                if var_name == "rainx_clim":  # from previous extreme < 2 to ari >=5
                    da.plot.contourf(
                        ax=ax,
                        transform=conf.plot_proj,
                        levels=levels,
                        colors=colors,  # previously white
                        alpha=1,
                        extend="both",
                        add_labels=False,
                        add_colorbar=False,
                    )
                p.colorbar.ax.set_title(
                    units,
                    pad=20 * fig_scale,
                    fontsize=10 * fig_scale,
                )
                if var_name in ["wpd", "ppv"]:
                    ax.annotate(
                        "$^{*}$Philippine landmass only",
                        xy=(5, -30),
                        xycoords="axes points",
                        fontsize=8 * fig_scale,
                        color=text_color,
                    )
                    ax.annotate(
                        plt_annotation,
                        xy=(5, -40),
                        xycoords="axes points",
                        fontsize=8 * fig_scale,
                    )
                else:
                    ax.annotate(
                        plt_annotation,
                        xy=(5, -30),
                        xycoords="axes points",
                        fontsize=8 * fig_scale,
                    )
                ax.annotate(
                    "observatory.ph",
                    xy=(10, 10),
                    xycoords="axes points",
                    fontsize=10 * fig_scale,
                    bbox=dict(boxstyle="square,pad=0.3", alpha=0.5),
                    alpha=0.5,
                )

                tt = int((t + 1) * hr_interval)
                file_prfx = "wrf"
                if "run" in da_name:
                    file_prfx += f"_{da_name}"
                out_file = (
                    out_dir
                    / f"{file_prfx}-{tt}hr_{var_name}_{init_dt.strftime('%Y-%m-%d_%H')}PHT.png"
                )
                fig.savefig(out_file, bbox_inches="tight", dpi=100)
                plt.close("all")
