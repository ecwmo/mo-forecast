from const import trmm

""" da must be daily WRF data with 3 ens members """


def model_agreement(ds):
    trmm2 = trmm.sel(time=(trmm.time.dt.month == ds.time.dt.month)).isel(time=0)

    _model_exceed = ds.where(ds < trmm2, 1)
    _model_counts = _model_exceed.where(_model_exceed == 1).sum("ens")

    return _model_counts
