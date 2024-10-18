import getopt
import os
import sys
from pathlib import Path

import pandas as pd
from config import Config
from extract_acenergy import extract_acenergy
from extract_points import extract_points
from extract_points_for_validation import extract_points_for_validation
from helpers.wrfpost import create_hour_ds, create_interval_ds, save_as_netcdf
from netCDF4 import Dataset
from plot_ari_maps import plot_ari
from plot_ecw_web_maps import plot_web_maps
from plot_hi_gauge import plot_gauge
from plot_maps import plot_maps
from plot_ts import plot_timeseries
from plot_ts_acenergy import plot_ts_ace
from wrf import omp_get_max_threads, omp_set_num_threads

omp_set_num_threads(int(os.getenv("SLURM_NTASKS", 4)))
print(f"Using {omp_get_max_threads()} threads...")


def main(wrfin, out_dir):
    hr_ds = create_hour_ds(wrfin)  # create hourly dataset
    init_dt = pd.to_datetime(hr_ds.time.values[0])
    init_dt_str = f"{init_dt:%Y%m%d}"
    init_dt_str_hh = f"{init_dt:%H}"
    out_file = out_dir / f"nc/wrf_{init_dt:%Y-%m-%d_%H}.nc"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {out_file}")
    save_as_netcdf(hr_ds, out_file)
    day_ds = create_interval_ds(hr_ds, 24)  # create daily dataset
    day_ds_rain = day_ds.rain  # select daily dataset rain only
    out_file_rain = out_dir / f"nc/wrf_{init_dt:%Y-%m-%d_%H}_rain.nc"
    save_as_netcdf(day_ds_rain, out_file_rain)
    three_hrly_ds = create_interval_ds(hr_ds, 3)  # create three hourly dataset

    # region output
    print("Creating maps...")
    _out_dir = out_dir / f"maps/24hrly/{init_dt_str}/{init_dt_str_hh}"
    _out_dir.mkdir(parents=True, exist_ok=True)
    plot_maps(day_ds, _out_dir)
    _out_dir = out_dir / f"maps/3hrly/{init_dt_str}/{init_dt_str_hh}"
    _out_dir.mkdir(parents=True, exist_ok=True)
    plot_maps(three_hrly_ds, _out_dir)

    _out_dir = out_dir / "timeseries/img"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating ts plot...")
    plot_timeseries(hr_ds, _out_dir)

    _out_dir = out_dir / "web/maps"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating web maps...")
    plot_web_maps(day_ds, _out_dir)

    _out_dir = out_dir / "web/json"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating summary...")
    extract_points({"hr": hr_ds, "day": day_ds}, _out_dir)

    _out_dir = out_dir / f"hi_gauge/img/{init_dt.strftime('%Y%m%d%H')}"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating HI gauge plots...")
    plot_gauge(hr_ds, _out_dir)

    _out_dir = out_dir / f"ari/img/{init_dt.strftime('%Y%m%d%H')}"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating ARI plots...")
    plot_ari(day_ds, _out_dir)

    _out_dir = out_dir / "acenergy/csv"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating summary for AC Energy solar farms...")
    extract_acenergy(hr_ds, _out_dir)

    _out_dir = out_dir / "acenergy/img"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating ts plot for AC Energy solar farms...")
    plot_ts_ace(hr_ds, _out_dir)

    _out_dir = out_dir / "stations_fcst_for_validation/json"
    _out_dir.mkdir(parents=True, exist_ok=True)
    print("Creating summary for stations for validation...")
    extract_points_for_validation({"hr": hr_ds, "day": day_ds}, _out_dir)
    # endregion output


if __name__ == "__main__":
    wrf_out = ""
    out_dir = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "odir="])
    except getopt.GetoptError:
        print("test.py -i <wrf_out file> -o <output dir>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("test.py -i <wrf_out file> -o <output dir>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            wrf_out = arg
        elif opt in ("-o", "--odir"):
            out_dir = Path(arg)
            out_dir.mkdir(parents=True, exist_ok=True)

    conf = Config()
    wrfin = {f"ens{i}": Dataset(d / wrf_out) for i, d in enumerate(conf.wrf_dirs)}
    main(wrfin, out_dir)
