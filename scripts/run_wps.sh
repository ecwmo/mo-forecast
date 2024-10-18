#!/usr/bin/env bash

source "$SCRIPT_DIR/set_env_wrf.run.sh"

cd "${WRF_MAINDIR}/WPS" || exit

mkdir "$NAMELIST_SUFF"

WPS_START_DATE="${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}:00:00"
WPS_END_DATE="${FCST_YY2}-${FCST_MM2}-${FCST_DD2}_${FCST_ZZ2}:00:00"
sed -i "4s/.*/\ start_date = '${WPS_START_DATE}','${WPS_START_DATE}',/" "namelist.wps_$NAMELIST_SUFF"
sed -i "5s/.*/\ end_date   = '${WPS_END_DATE}','${WPS_END_DATE}',/" "namelist.wps_$NAMELIST_SUFF"

rm -f namelist.wps
ln -s "namelist.wps_$NAMELIST_SUFF" namelist.wps

rm -f "$NAMELIST_SUFF"/geo_em*
echo "************************"
echo "*   start of geogrid   *"
echo "************************"
srun -n 1 ./geogrid.exe >&log.geogrid &
tail --pid=$! -f log.geogrid
if ! tail -n 5 "log.geogrid" | grep -q "Successful"; then
  echo "geogrid" >>"$ERROR_FILE"
  echo "************************"
  echo "*     geogrid error    *"
  echo "************************"
  exit 1
fi
echo "************************"
echo "*    end of geogrid    *"
echo "************************"

rm -f GRIBFILE.*
# Update GFS link
rm -f "$WPS_GFSDIR"/*.grb
ln -sf "$GFS_DIR"/*.grb "$WPS_GFSDIR"/.
./link_grib.csh "$WPS_GFSDIR"/*.grb
#ln -s ungrib/Variable_Tables/Vtable.GFS Vtable #new line added; one time use

rm -f FILE:*
echo "************************"
echo "*   start of ungrib    *"
echo "************************"
srun -n 1 ./ungrib.exe >&log.ungrib &
tail --pid=$! -f log.ungrib
if ! tail -n 5 "log.ungrib" | grep -q "Successful"; then
  echo "ungrib" >>"$ERROR_FILE"
  echo "************************"
  echo "*     ungrib error     *"
  echo "************************"
  exit 1
fi
echo "************************"
echo "*    end of ungrib     *"
echo "************************"

rm -f "$NAMELIST_SUFF"/met_em.d0*
echo "************************"
echo "*   start of metgrid   *"
echo "************************"
srun ./metgrid.exe >&log.metgrid &
tail --pid=$! -f log.metgrid
if ! tail -n 5 "log.metgrid" | grep -q "Successful"; then
  echo "metgrid" >>"$ERROR_FILE"
  echo "************************"
  echo "*     metgrid error    *"
  echo "************************"
  exit 1
fi
echo "************************"
echo "*    end of metgrid    *"
echo "************************"

rm -f geo_em*
rm -f FILE:*
rm -f GRIB*
