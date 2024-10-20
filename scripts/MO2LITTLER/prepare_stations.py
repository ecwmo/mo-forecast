from dotenv import dotenv_values
from pathlib import Path
import pandas as pd
import numpy as np

config = dotenv_values("./.env")

df_station = pd.read_csv(Path(config["IN_DIR"]) / "coordinates.csv", delimiter="\t")
stations = df_station["Station ID"].tolist()
j = 1
stationlist = []
for i in stations:
    print(i)
    # i='SM_Marikina'
    raw = pd.read_csv(
        Path(config["IN_DIR"])
        / f"stn_obs_24hr_{config['YYYYMMDD']}_{config['HH_INIT']}.csv"
    )
    raw["name"] = raw["name"].str.replace(" ", "_")
    raw["name"] = raw["name"].str.replace(".", "")
    raw = raw.loc[raw["name"] == f"{i}"]
    if len(raw) == 0:  # if there are no station values
        continue
    raw["date"] = pd.to_datetime(raw["timestamp"])
    utc = raw["date"].dt.tz_convert("Etc/GMT+1")
    raw["Stn.No."] = j
    raw["Year"] = utc.dt.year
    raw["Month"] = utc.dt.month
    raw["Day"] = utc.dt.day
    raw["HH"] = utc.dt.hour
    raw["T"] = raw["temp"]
    raw["W"] = raw["wspd"]
    raw["WD"] = raw["wdir"]
    raw["u"] = -raw["W"] * np.cos(2 * np.pi / 360 * (90 - raw["WD"]))
    raw["v"] = -raw["W"] * np.sin(2 * np.pi / 360 * (90 - raw["WD"]))
    raw["p"] = raw["pres"]
    raw["R"] = raw["rr"]
    raw["RH"] = raw["rh"]
    raw["is_sound"] = ".false."
    clean = raw[
        [
            "Stn.No.",
            "Year",
            "Month",
            "Day",
            "HH",
            "T",
            "W",
            "WD",
            "u",
            "v",
            "p",
            "R",
            "RH",
            "is_sound",
        ]
    ]
    cleans = clean.sort_values(by=["Day", "HH"])
    cleans.to_csv(Path(config["OUT_DIR"]) / f"{i}.csv", index=False)
    j = j + 1

    for rn in raw["name"].unique():
        stationlist.append(rn)
# Get list of stations with data
st = pd.DataFrame(stationlist, columns=["Station ID"])
d_station_new = pd.merge(st, df_station, on="Station ID")
# Create new csv of station with coordinates for convert.exe
d_station_new.to_csv(
    Path(config["IN_DIR"]) / "MO2LITTLER/coordinates_data.csv", index=False
)
