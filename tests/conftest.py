import pytest
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from typing import Optional


@pytest.fixture
def bash_dir(request):
    return Path(request.config.rootdir) / "scripts"


@pytest.fixture
def run_script(bash_dir):
    def r(script_path: str):
        result = subprocess.run(
            ["bash", str(bash_dir / script_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result

    return r


@pytest.fixture(autouse=True)
def cron_env(request, tmp_path, monkeypatch, bash_dir):
    orig_path = os.getenv("PATH")
    new_path = request.config.rootdir + "/tests/bin"
    if orig_path is not None:
        new_path += ":" + orig_path
    monkeypatch.setenv("PATH", new_path)

    monkeypatch.setenv("SCRIPT_DIR", str(bash_dir))

    maindir = tmp_path
    monkeypatch.setenv("MAINDIR", str(maindir))

    tmpdir = maindir / ".tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TEMP_DIR", str(tmpdir))

    gfsdir = maindir / "input/gfs_files"
    gfsdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GFS_BASE_DIR", str(gfsdir))

    wrf_maindir = maindir / "model"
    monkeypatch.setenv("WRF_MAINDIR", str(wrf_maindir))
    wrf_realdir = wrf_maindir / "WRF"
    wrf_realdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WRF_REALDIR", str(wrf_realdir))
    wps_maindir = wrf_maindir / "WPS"
    wps_maindir.mkdir(parents=True, exist_ok=True)
    wps_gfsdir = wps_maindir / "GFS"
    wps_gfsdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WPS_GFSDIR", str(wps_gfsdir))

    monkeypatch.setenv("WRF_MODE", "test")

    ndays = 5
    monkeypatch.setenv("WRF_FCST_DAYS", f"{ndays}")

    namelist_suff = "suff"
    monkeypatch.setenv("NAMELIST_SUFF", namelist_suff)
    monkeypatch.setenv("WPS_NAMELIST_SUFF", namelist_suff)
    (wps_maindir / f"namelist.wps_{namelist_suff}").write_text("\n" * 5)

    namelist_run = "run1"
    monkeypatch.setenv("NAMELIST_RUN", namelist_run)
    (wrf_realdir / f"namelist.input_{namelist_run}").write_text("\n" * 15)

    link_grib_csh_text = """#!/bin/csh

    echo running link_grib.csh
    set pattern = $argv[1]
    set files = (`ls $pattern`)

    foreach file ($files)
        echo $file
    end
    """
    fname = str(wps_maindir / "link_grib.csh")
    with open(fname, "w") as file:
        file.write(link_grib_csh_text)
    os.chmod(fname, 0o755)


@pytest.fixture
def set_date_env(monkeypatch):
    def d(ts: datetime):
        _ndays = os.getenv("WRF_FCST_DAYS")
        ndays = 3
        if _ndays is not None:
            ndays = int(_ndays)
        ts2 = ts + timedelta(days=ndays)
        monkeypatch.setenv("FCST_YYYYMMDD", f"{ts:%Y%m%d}")
        monkeypatch.setenv("FCST_YY", f"{ts:%Y}")
        monkeypatch.setenv("FCST_MM", f"{ts:%m}")
        monkeypatch.setenv("FCST_DD", f"{ts:%d}")
        monkeypatch.setenv("FCST_ZZ", f"{ts:%H}")
        monkeypatch.setenv("FCST_YY2", f"{ts2:%Y}")
        monkeypatch.setenv("FCST_MM2", f"{ts2:%m}")
        monkeypatch.setenv("FCST_DD2", f"{ts2:%d}")
        monkeypatch.setenv("FCST_ZZ2", f"{ts2:%H}")

        gfs_basedir = os.getenv("GFS_BASE_DIR")
        if gfs_basedir is not None:
            monkeypatch.setenv("GFS_DIR", f"{gfs_basedir}/{ts:%Y%m%d/%H}")

    return d


@pytest.fixture
def create_met_ems():
    def c(nfiles: Optional[int] = None):
        if nfiles is None:
            _ndays = os.getenv("WRF_FCST_DAYS")
            ndays = 3
            if _ndays is not None:
                ndays = int(_ndays)
            nfiles = ndays * 24 + 1

        wrf_maindir = os.getenv("WRF_MAINDIR")
        wps_namelist_suff = os.getenv("WPS_NAMELIST_SUFF")

        if (wrf_maindir is None) or (wps_namelist_suff is None):
            return

        met_em_dir = Path(wrf_maindir) / f"WPS/{wps_namelist_suff}"
        met_em_dir.mkdir(parents=True, exist_ok=True)

        for i in range(nfiles):
            f = met_em_dir / f"met_em.d01.{i:04d}:00:00.nc"
            f.touch()

    return c
