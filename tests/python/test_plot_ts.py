import pytest
from pathlib import Path

from plot_ts import plot_timeseries
import xarray as xr


def test_plot_timeseries(wrfnc: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.exists()

    hr_ds = xr.open_dataset(wrfnc)
    assert isinstance(hr_ds, xr.Dataset)
    
    plot_timeseries(hr_ds, out_dir)
    assert len(list(out_dir.glob("*.png"))) > 0
