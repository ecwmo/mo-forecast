from pathlib import Path
import pandas as pd
import xarray as xr

from config import Config


def rain_chance_str(val):
    if val == 0:
        return "NoChance"
    elif val <= 1 / 3:
        return "Low"
    elif val <= 2 / 3:
        return "Medium"
    else:
        return "High"


def extract_points(ds_dict, out_dir):
    conf = Config()
    site_df = pd.read_csv(conf.script_dir / "python/resources/csv/cities.csv")

    init_dt = pd.to_datetime(
        list(ds_dict.values())[0].time.values[0], utc=True
    ).astimezone(conf.tz)

    new_ds_dict = {}
    for k, ds in ds_dict.items():
        _ds = []

        # region wpd
        print("Processing wpd...")

        da = ds["wpd"].mean("ens")
        da.name = "wndPow"
        _ds.append(da)
        da = ds["wpd"].min("ens")
        da.name = "wndPowMin"
        _ds.append(da)
        da = ds["wpd"].max("ens")
        da.name = "wndPowMax"
        _ds.append(da)
        # endregion wpd

        # region ppv
        print("Processing ppv...")

        da = ds["ppv"].mean("ens")
        da.name = "solPow"
        _ds.append(da)
        da = ds["ppv"].min("ens")
        da.name = "solPowMin"
        _ds.append(da)
        da = ds["ppv"].max("ens")
        da.name = "solPowMax"
        _ds.append(da)
        # endregion ppv

        # region temp
        print("Processing temperature...")
        da = ds["temp"].mean("ens")
        da.name = "temp"
        _ds.append(da)
        da = ds["temp"].min("ens")
        da.name = "tempMin"
        _ds.append(da)
        da = ds["temp"].max("ens")
        da.name = "tempMax"
        _ds.append(da)
        # endregion temp

        # region wind
        print("Processing wind...")
        u = ds["u_850hPa"]
        v = ds["v_850hPa"]
        _da = u.copy()
        _da.values = (u.values**2 + v.values**2) ** 0.5
        _da.values = _da.values * 3.6  # convert to kph

        da = _da.mean("ens")
        da.name = "wspd"
        _ds.append(da)
        da = _da.min("ens")
        da.name = "wspdMin"
        _ds.append(da)
        da = _da.max("ens")
        da.name = "wspdMax"
        _ds.append(da)
        # endregion wind

        # region rain
        print("Processing chance of rain...")
        _da = ds["rain"]

        da = _da.mean("ens")
        da.name = "rain"
        _ds.append(da)
        no_rain = _da <= 0.2
        _da = _da.where(no_rain, 1)
        _da = _da.where(~no_rain, 0)
        da = _da.mean("ens")
        da.name = "rainChance"
        _ds.append(da)
        # endregion rain

        # region relative humidity
        print("Processing relative humidiy...")
        _da = ds["rh"]

        da = _da.mean("ens")
        da.name = "rh"
        _ds.append(da)
        # endregion relative humidity

        # region heat index
        print("Processing heat index...")
        _da = ds["hi"]

        da = _da.mean("ens")
        da.name = "hi"
        _ds.append(da)
        # endregion heat index

        new_ds_dict[k] = xr.merge(_ds)

    out_df = []
    for idx, r in site_df.iterrows():
        df_dict = {}
        for k, ds in new_ds_dict.items():
            _df = ds.sel(lon=r["lon"], lat=r["lat"], method="nearest").to_dataframe()
            _df.drop(columns=["lon", "lat"], inplace=True)
            _df.reset_index(inplace=True)
            _df.rename(columns={"time": "timestamp"}, inplace=True)
            _df["timestamp"] = (
                _df["timestamp"]
                .dt.tz_localize("UTC")
                .dt.tz_convert(conf.tz)
                .dt.strftime("%Y-%m-%dT%H:00:00")
            )
            _df["rainChanceStr"] = _df["rainChance"].apply(rain_chance_str)
            df_dict[k] = _df.to_dict(orient="records")
        _out_df = r.to_dict()
        _out_df["id"] = r["name"]
        _out_df["forecast"] = df_dict
        out_df.append(_out_df)

    out_df = pd.DataFrame(out_df)
    out_file = Path(out_dir) / f"forecast_{init_dt.strftime('%Y-%m-%d_%H')}PHT.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.set_index("id").to_json(
        out_file, orient="index", indent=2, double_precision=2
    )
