#!/usr/bin/env bash

source "$SCRIPT_DIR/set_env_wrf.run.sh"

cd "${WRF_MAINDIR}/WPS" || exit

mkdir -p "$NAMELIST_SUFF/gfs"

WPS_NAMELIST_GFS="namelist.wps_gfs_$NAMELIST_SUFF"

WPS_START_DATE="${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}:00:00"
WPS_END_DATE="${FCST_YY2}-${FCST_MM2}-${FCST_DD2}_${FCST_ZZ2}:00:00"
sed -i "4s/.*/\ start_date = '${WPS_START_DATE}','${WPS_START_DATE}',/" "$WPS_NAMELIST_GFS"
sed -i "5s/.*/\ end_date   = '${WPS_END_DATE}','${WPS_END_DATE}',/" "$WPS_NAMELIST_GFS"

ln -sf "$WPS_NAMELIST_GFS" namelist.wps

rm -f "$NAMELIST_SUFF"/geo_em*
echo "------------------"
echo " Start of Geogrid "
echo "------------------"
srun -n 1 ./geogrid.exe >&log.geogrid &
tail --pid=$! -f log.geogrid
echo "------------------"
echo " End of Geogrid "
echo "------------------"

rm -f GRIBFILE.*
# Update GFS link
rm -f "$WPS_GFSDIR"/*.grb
ln -sf "$GFS_DIR"/*.grb "$WPS_GFSDIR"/.
./link_grib.csh "$WPS_GFSDIR"/*.grb
#ln -s ungrib/Variable_Tables/Vtable.GFS Vtable #new line added; one time use

rm -f GFS:* 
echo "------------------"
echo " Start of Ungrib "
echo "------------------"
srun -n 1 ./ungrib.exe >&log.ungrib &
tail --pid=$! -f log.ungrib
echo "------------------"
echo " End of Ungrib "
echo "------------------"

rm -f "$NAMELIST_SUFF"/gfs/met_em.d0*
echo " Start of Metgrid "
echo "------------------"
srun ./metgrid.exe >&log.metgrid &
tail --pid $! -f log.metgrid
echo "------------------"
echo " End of Metgrid "
echo "------------------"


mkdir -p "$NAMELIST_SUFF/ecmwf"

WPS_NAMELIST_ECMWF="namelist.wps_ecmwf_$NAMELIST_SUFF"
sed -i "4s/.*/\ start_date = '${WPS_START_DATE}','${WPS_START_DATE}',/" "$WPS_NAMELIST_ECMWF"
sed -i "5s/.*/\ end_date   = '${WPS_END_DATE}','${WPS_END_DATE}',/" "$WPS_NAMELIST_ECMWF"

ln -sf "$WPS_NAMELIST_ECMWF" namelist.wps


# Update ECMWF link
rm -f "$WPS_ECMWFDIR"/*.nc
ln -sf "$ECMWF_NC_DIR"/*.nc "$WPS_ECMWFDIR"/.

rm -f ECMWF:*
echo "------------------"
echo " Start of Ungrib "
echo "------------------"
cd "$SCRIPT_DIR/python" || exit
$PYTHON convert_nc_interfile.py -i "$WPS_ECMWFDIR" -o "${WRF_MAINDIR}/WPS" "${FCST_YYYYMMDD}"
cd "${WRF_MAINDIR}/WPS" || exit
echo "------------------"
echo " End of Ungrib "
echo "------------------"

rm -f "$NAMELIST_SUFF"/ecmwf/met_em.d0*
echo " Start of Metgrid "
echo "------------------"
srun ./metgrid.exe >&log.metgrid &
tail --pid $! -f log.metgrid
echo "------------------"
echo " End of Metgrid "
echo "------------------"


# cleanup
rm -f geo_em*
rm -f GFS:*
rm -f ECMWF:*
rm -f GRIB*

