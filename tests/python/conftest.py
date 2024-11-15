import pytest
from pathlib import Path


@pytest.fixture
def data_dir(request):
    return Path(request.config.rootdir) / "tests/python/data"


@pytest.fixture
def wrfout(data_dir: Path):
    file = data_dir / "wrfout_d01_subset"
    if file.is_file():
        return file
    return None


@pytest.fixture
def wrfnc(data_dir: Path):
    file = data_dir / "wrf.nc"
    if file.is_file():
        return file
    return None


@pytest.fixture
def ecmwfnc(data_dir: Path):
    file = data_dir / "ecmwf_sample.nc"
    if file.is_file():
        return file
    return None
