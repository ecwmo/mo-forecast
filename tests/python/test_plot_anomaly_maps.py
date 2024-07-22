from plot_anomaly_maps import open_obs


def test_open_obs():
    obs = open_obs()
    assert obs is not None
    assert obs.temp is not None
    assert obs.temp.sum() > 0
    assert obs.rain is not None
    assert obs.rain.sum() > 0
    assert obs.month is not None
    assert len(obs.month) == 12
    assert obs.lat is not None
    assert len(obs.lat) == 415
    assert obs.lon is not None
    assert len(obs.lon) == 320
