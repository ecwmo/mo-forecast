import os
from pathlib import Path
from datetime import datetime, timedelta
import pytest


@pytest.mark.parametrize(
    "err_on, has_error",
    [
        (None, False),
        ("geogrid", True),
        ("ungrib", True),
        ("metgrid", True),
        ("metgrid:2", True),
        ("foo", False),
        ("foo:1", False),
    ],
)
def test_wps(monkeypatch, run_script, set_date_env, err_on, has_error):
    ts = datetime(2024, 6, 3, 12)
    set_date_env(ts)

    monkeypatch.setenv("SLURM_NTASKS", "4")

    tmpdir = os.getenv("TEMP_DIR")
    assert tmpdir is not None
    assert Path(tmpdir).exists()
    err_file = Path(tmpdir) / "error.txt"

    monkeypatch.setenv("ERROR_FILE", str(err_file))

    if err_on is not None:
        monkeypatch.setenv("TEST_ERR_ON", err_on)
    if has_error:
        with pytest.raises(Exception):
            run_script("run_wps.sh")
        assert err_file.exists()
    else:
        result = run_script("run_wps.sh")
        assert result.returncode == 0
        assert not err_file.exists()
        assert_wps_namelist(ts)


@pytest.mark.parametrize(
    "err_on, has_error",
    [
        (None, False),
        ("real", True),
        ("real:2", True),
        ("wrf", True),
        ("wrf:3", True),
        ("foo", False),
        ("foo:1", False),
    ],
)
def test_wrf(monkeypatch, run_script, set_date_env, create_met_ems, err_on, has_error):
    ts = datetime(2024, 6, 3, 12)
    set_date_env(ts)

    monkeypatch.setenv("SLURM_NTASKS", "4")

    tmpdir = os.getenv("TEMP_DIR")
    assert tmpdir is not None
    assert Path(tmpdir).exists()
    err_file = Path(tmpdir) / "error.txt"

    monkeypatch.setenv("ERROR_FILE", str(err_file))

    if err_on is not None:
        monkeypatch.setenv("TEST_ERR_ON", err_on)
    if has_error:
        with pytest.raises(Exception):
            create_met_ems()
            run_script("run_wrf.sh")
        assert err_file.exists()
    else:
        create_met_ems()
        result = run_script("run_wrf.sh")
        assert result.returncode == 0
        assert not err_file.exists()
        assert_wrf_namelist(ts)


def assert_wps_namelist(ts: datetime):
    wrf_maindir = os.getenv("WRF_MAINDIR")
    namelist_suff = os.getenv("NAMELIST_SUFF")
    assert wrf_maindir is not None
    namelist = Path(wrf_maindir) / f"WPS/namelist.wps_{namelist_suff}"
    assert namelist.exists()

    with open(str(namelist), "r") as file:
        lines = file.readlines()

    assert len(lines) >= 5
    assert f"{ts:%Y-%m-%d_%H:00:00}" in lines[3].strip()

    ndays = os.getenv("WRF_FCST_DAYS")
    assert ndays is not None
    ts2 = ts + timedelta(days=int(ndays))
    assert f"{ts2:%Y-%m-%d_%H:00:00}" in lines[4].strip()


def assert_wrf_namelist(ts: datetime):
    wrf_realdir = os.getenv("WRF_REALDIR")
    namelist_run = os.getenv("NAMELIST_RUN")
    assert wrf_realdir is not None
    namelist = Path(wrf_realdir) / f"namelist.input_{namelist_run}"
    assert namelist.exists()

    with open(str(namelist), "r") as file:
        lines = file.readlines()

    assert len(lines) >= 15

    ndays = os.getenv("WRF_FCST_DAYS")
    assert ndays is not None

    assert ndays in lines[1].strip()
    assert f"{ts:%Y}" in lines[5].strip()
    assert f"{ts:%m}" in lines[6].strip()
    assert f"{ts:%d}" in lines[7].strip()
    assert f"{ts:%H}" in lines[8].strip()
    ts2 = ts + timedelta(days=int(ndays))
    assert f"{ts2:%Y}" in lines[11].strip()
    assert f"{ts2:%m}" in lines[12].strip()
    assert f"{ts2:%d}" in lines[13].strip()
    assert f"{ts2:%H}" in lines[14].strip()
