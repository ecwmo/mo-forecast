import numpy as np
import pywinter.winter as pyw
import xarray as xr 
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units
import datetime as dt
import argparse
import os
import glob


def nctointerfile(in_dir, out_dir, day_string):
    day = dt.datetime.strptime(day_string, '%Y%m%d').date()
    filepath = glob.glob(os.path.join(os.path.dirname(in_dir), "ECMWF*.nc"))
    for file in filepath:
        hour = np.abs(int(os.path.basename(file)[5:][:3]))
        ### Paths and Files
        ecmwf = xr.open_dataset(f'{file}').isel(
            height = 0, height_2 = 0, height_3 = 0).drop(['height', 'height_2', 'height_3']).sortby('lat')
        ecmwf = ecmwf.sel(lat=slice(-25,30), lon=slice(60,180)).fillna(-9999.)
        ecmwf = ecmwf.assign(rh2 = (mpcalc.relative_humidity_from_dewpoint(ecmwf['2t'] * units.kelvin, ecmwf['2d'] * units.kelvin)).metpy.dequantify())
        ecmwf = ecmwf.assign(q2 = (mpcalc.specific_humidity_from_dewpoint(ecmwf['sp'] * units.Pa, ecmwf['2d'] * units.kelvin)).metpy.dequantify())

        layers = ['000007','007028','028100','100200'] 
        st3 = ecmwf['stl3']
        sm3 = ecmwf['swvl3']
        st4 = ecmwf['stl4']
        sm4 = ecmwf['swvl4']
        st1 = ecmwf['st']
        st2 = ecmwf['stl2']
        stl = xr.concat([st1, st2, st3, st4], pd.Index(['000007','007028','028100','100200'], name="layer"))
        ecmwf = ecmwf.assign(tsl = stl)

        sm1 = ecmwf['swvl1']
        sm2 = ecmwf['swvl2']
        sml = xr.concat([sm1, sm2, sm3, sm4], pd.Index(['000007','007028','028100','100200'], name="layer"))
        ecmwf = ecmwf.assign(mrlsl = sml)

        ecmwf = ecmwf.isel(depth = 0, depth_2 = 0, depth_3 = 0, depth_4 = 0, depth_5 = 0).drop(
            ['depth', 'depth_2', 'depth_3', 'depth_4', 'depth_5'])

        ### Prepare Geo data (Geo0: Cylindrical Equidistant)
        # this assumes the ecmwf file has the same baseline info as the others 
        lat = ecmwf.lat.values[:]  
        lon = ecmwf.lon.values[:] # need SW corner latitude and longitude
        dlat = np.abs(lat[1] - lat[0])
        dlon = np.abs(lon[1] - lon[0]) # need latitude and longitude increments 
        geo = pyw.Geo0(lat[0], lon[0], dlat, dlon)

        ### Loop through all the available timesteps in the files (follow 6-hourly resolution of ecmwf file)
        for t in ecmwf.time.values:
            ## Read 2D data and create 2D fields with pyw.V2d() function
            # Surface Pressure
            ps = ecmwf.sp.sel(time=t).values  # take at 1st timestep and convert to hPa
            PSFC = pyw.V2d('PSFC',ps)
            # Skin Temperature
            ts = ecmwf.skt.sel(time=t).values
            SKINTEMP = pyw.V2d('SKINTEMP',ts)
            # Mean Sea-level Pressure
            mslp = ecmwf.msl.sel(time=t).values
            PMSL = pyw.V2d('PMSL',mslp)
            # 2M Air Temperature
            tas = ecmwf['2t'].sel(time=t)[:,:].values
            TT2M = pyw.V2d('TT',tas)
            # 2M Ralative Humidity
            huss = ecmwf.q2.sel(time=t)[:,:].values # no qas ?
            SPECHUMD2M = pyw.V2d('SPECHUMD',huss)
            # 10M wind u-component
            uas = ecmwf['10u'].sel(time=t)[:,:].values
            UU10M = pyw.V2d('UU',uas)
            # 10M wind v-component
            vas = ecmwf['10v'].sel(time=t)[:,:].values
            VV10M = pyw.V2d('VV',vas)
            # Land-sea mask (0=water, 1=land)
            landmask = ecmwf['lsm'].sel(time=t)[:,:].values
            LANDSEA = pyw.V2d('LANDSEA',landmask)

            ## Prepare 3D data and create 3D isobaric fields with pyw.V3dp() function 
            plevs = ecmwf.plev.values
            # 3D Air Temperature 
            ta = ecmwf.t.sel(time=t).values
            TT = pyw.V3dp('TT',ta,plevs)
            # 3D Relative Humidity 
            rh = ecmwf.r.sel(time=t).values
            RH = pyw.V3dp('RH',rh,plevs)
            # 3D Specific Humidity 
            hus = ecmwf.q.sel(time=t).values
            SPECHUMD = pyw.V3dp('SPECHUMD',hus,plevs)
            # 3D Wind u-component 
            ua = ecmwf.u.sel(time=t).values
            UU = pyw.V3dp('UU',ua,plevs)
            # 3D wind v-component 
            va = ecmwf.v.sel(time=t).values
            VV = pyw.V3dp('VV',va,plevs)
            # 3D geopotential height 
            hgt = ecmwf.gh.sel(time=t).values
            GHT = pyw.V3dp('GHT',hgt,plevs)
            
            #SOIL 
            # Select data 
            stsel = ecmwf.tsl.sel(time=t).values
            smsel = ecmwf.mrlsl.sel(time=t).values
            # Create fields 
            ST = pyw.Vsl('ST',stsel,layers)
            SM = pyw.Vsl('SM',smsel,layers)
            ### Create the intermediate file 
            #findate = t.strftime('%Y-%m-%d_%H')
            fields = [PSFC, SKINTEMP, PMSL, TT2M, SPECHUMD2M, UU10M, VV10M, TT, LANDSEA, RH, SPECHUMD, UU, VV, GHT, ST, SM]
            if hour in np.arange(0,24,3):
                if hour < 10:
                    pyw.cinter('ECMWF', f'{day}_0{hour}', geo, fields, out_dir)
                else:
                    pyw.cinter('ECMWF', f'{day}_{hour}', geo, fields, out_dir)
            elif hour in np.arange(24,48,3):
                hr = hour - 24
                if hr < 10:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=1)}_0{hr}', geo, fields, out_dir)
                else:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=1)}_{hr}', geo, fields, out_dir)
            elif hour in np.arange(48,72,3):
                hr = hour - 48
                if hr < 10:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=2)}_0{hr}', geo, fields, out_dir)
                else:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=2)}_{hr}', geo, fields, out_dir)
            elif hour in np.arange(72,96,3):
                hr = hour - 72
                if hr < 10:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=3)}_0{hr}', geo, fields, out_dir)
                else:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=3)}_{hr}', geo, fields, out_dir)
            elif hour in np.arange(96,120,3):
                hr = hour - 96
                if hr < 10:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=4)}_0{hr}', geo, fields, out_dir)
                else:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=4)}_{hr}', geo, fields, out_dir)
            elif hour in np.arange(120,144,3):
                hr = hour - 120
                if hr < 10:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=5)}_0{hr}', geo, fields, out_dir)
                else:
                    pyw.cinter('ECMWF', f'{day + dt.timedelta(days=5)}_{hr}', geo, fields, out_dir)          
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="A script for a new parameter value.")
    # Adding three positional arguments of type float
    parser.add_argument("in_dir", help="NETCDF file")
    parser.add_argument("out_dir", help="Folder to save interfile")
    parser.add_argument("day_string", help="Date in %Y%m%d format")
    
    # Parse the arguments
    args = parser.parse_args()
    
    nctointerfile(args.in_dir, args.out_dir, args.day_string)

