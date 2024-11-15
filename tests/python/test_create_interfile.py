import shutil

from create_interfile import nctointerfile

def test_nctointerfile(tmp_path, ecmwfnc):
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    day_string = "20240101"

    
    [shutil.copy(ecmwfnc, in_dir / f"ECMWF{h:03d}.nc") for h in range(0, 121, 6)]
    files = list(in_dir.glob("ECMWF*.nc"))

    nctointerfile(in_dir, out_dir, day_string)
    
    out_files = list(out_dir.glob("*"))

    assert len(out_files) == len(files)
