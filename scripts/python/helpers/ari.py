import xarray as xr
import salem
import xesmf as xe
from scipy.interpolate import interp1d

from config import Config

config = Config()
resource_dir = config.script_dir


def open_ref():
    _domain = (
        xr.open_dataset(resource_dir / "python/resources/nc/trmm_domain_regrid.nc")
        .rename({"precipitation": "rain", "lon": "longitude", "lat": "latitude"})
        .sel(longitude=slice(117.375, 126.375), latitude=slice(5.125, 18.875))
        .drop("time_bnds")
    )
    _domain = _domain.drop(("time", "rain"))

    return _domain


def open_ari():
    ari = [1.5, 2, 5, 10, 25, 30, 50, 100, 200, 500, 1000]
    _ds = xr.open_dataset(resource_dir / "python/resources/nc/PHIL_ARI.nc").rename(
        {"precip": "rain"}
    )
    _ds = _ds.assign_coords({"ari": ari})

    return _ds


def mask(ds):
    _mask = salem.read_shapefile(
        resource_dir / "python/resources/shp/PHL_adm0/PHL_adm0.shp"
    )
    _ds_mask = ds.salem.roi(shape=_mask)

    return _ds_mask


def regrid(ds, _ds_out, method="bilinear"):
    return xe.Regridder(ds, _ds_out, method)(ds)


def interpd(_newx, _vals, _idxs):
    return interp1d(_vals, _idxs, bounds_error=False, fill_value="extrapolate")(_newx)


def ARIinterp(_newx):
    _dat = open_ari()

    _ari = [1.5, 2, 5, 10, 25, 30, 50, 100, 200, 500, 1000]
    _idx = xr.DataArray(_ari, dims=["ari"]).broadcast_like(_dat).to_dataset(name="rain")

    _ari_interpd = xr.apply_ufunc(
        interpd,
        _newx,
        _dat,
        _idx,
        input_core_dims=[["time"], ["ari"], ["ari"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
    )
    return _ari_interpd


def run_interp(ds):
    _ref = open_ref()
    _wrf_ari = ARIinterp(ds)
    _wrf_ari = _wrf_ari.rename({"longitude": "lon", "latitude": "lat"})

    return _wrf_ari
