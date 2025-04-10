# %%
# Import Libraries
import pandas as pd
import numpy as np
import xarray as xr
import time
from datetime import datetime, timedelta
import os
from config import Config


conf = Config()
resources_dir = conf.script_dir / "python/resources/nc"
nc_dir = conf.data_dir / "daily/nc"
inp_dir = os.getenv("WRF_SERVER_DIR")
out_dir = conf.data_dir / "daily"


def get_nrt(ds):
    ds
    return ds.isel(time=slice(6))[["rain", "temp"]].mean("ens")


def save_daily(day_offset=1):
    # Input: (day_offset) - number of days relative to day of processing which is always Yesterday (n=1). Change for earlier days.
    start = time.time()
    # Get yesterday"s dataset from Dugong and Arowana output nc
    _day = datetime.today().date() - timedelta(days=day_offset)
    print(f"Processing output from {_day}...")
    try:
        _arowana = xr.open_mfdataset(
            f"{inp_dir}/output.arowana/nc/wrf_{_day}_??.nc",
            concat_dim="time",
            combine="nested",
            preprocess=get_nrt,
            chunks="auto",
            parallel=True,
        )

        _dugong = xr.open_mfdataset(
            f"{inp_dir}/output.dugong/nc/wrf_{_day}_??.nc",
            concat_dim="time",
            combine="nested",
            preprocess=get_nrt,
            chunks="auto",
            parallel=True,
        )
        _combined = _arowana.combine_first(_dugong)
    except OSError:
        print(f'Files don"t exist for {_day}')
        end = time.time()
        elapsed = end - start

        print(
            f"▮▮▮ Elapsed time in real time : {datetime.strftime(datetime.utcfromtimestamp(elapsed), '%H:%M:%S')}"
        )
        return

    # Check for missing hours
    _complete = list(np.arange(0, 24))
    _hours = _combined.time.dt.hour.values
    _missing = [item for item in _complete if item not in _hours]
    if len(_missing) == 0:  # For complete data
        print(f"Dataset is complete\nSaving Dataset for {_day}...")
        _final = _combined.resample({"time": "D"}).mean()
        _final["rain"] = (
            _combined["rain"].resample({"time": "D"}).sum()
        )  # Since rain should be accumulated and not mean
        _final.load().to_netcdf(out_dir / f"nc/wrf_{_day}.nc")

    # For incomplete and partially complete Datasets
    if len(_missing) > 4:  # For incomplete data
        _ref_dates = pd.date_range(start=f"{_day}", periods=24, freq="H")
        _act_dates = pd.Index(_combined.time.values)
        _missing_hours = _ref_dates.difference(_act_dates)
        _missing_hours.to_series().to_csv(out_dir + f"txt/Missing_{_day}.txt")
        print(f"Missing Hours: {len(_missing)}\nNot Saving Dataset...")

    if (
        len(_missing) < 4 and len(_missing) > 0
    ):  # For partially complete data (<4 hrs missing)
        print(f"Missing Hours: {len(_missing)}\nSaving Dataset for {_day}...")
        # Logging of missing data
        _ref_dates = pd.date_range(start=f"{_day}", periods=24, freq="H")
        _act_dates = pd.Index(_combined.time.values)
        _missing_hours = _ref_dates.difference(_act_dates)
        _missing_hours.to_series().to_csv(
            out_dir + f"txt/P_Missing_{_day}.txt"
        )  # Still logs partially missing days
        # Process and Save Dataset
        _final = _combined.resample({"time": "D"}).mean()
        _final["rain"] = (
            _combined["rain"].resample({"time": "D"}).sum()
        )  # Since rain should be accumulated and not mean
        _final.load().to_netcdf(out_dir / f"nc/wrf_{_day}.nc")

    end = time.time()
    elapsed = end - start
    print(
        f"▮▮▮ Elapsed time in real time : {datetime.strftime(datetime.utcfromtimestamp(elapsed), '%H:%M:%S')}"
    )


if __name__ == "__main__":
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "img"
    txt_dir = out_dir / "txt"
    nc_dir = out_dir / "nc"

    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    nc_dir.mkdir(parents=True, exist_ok=True)

    save_daily()
