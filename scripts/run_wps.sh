#!/usr/bin/env bash

generate_namelist() {
  local NAMELIST_FILE="$1"

  sed -i "4s/.*/\ start_date = '${WPS_START_DATE}','${WPS_START_DATE}',/" "${NAMELIST_FILE}"
  sed -i "5s/.*/\ end_date   = '${WPS_END_DATE}','${WPS_END_DATE}',/" "${NAMELIST_FILE}"

  rm -f namelist.wps
  ln -s "${NAMELIST_FILE}" namelist.wps
}

run_geogrid() {
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
}

run_ungrib() {
  local GRIB_DIR="$1"
  local INTER_FILE="$2"

  ./link_grib.csh "${GRIB_DIR}"/*.grb
  #ln -s ungrib/Variable_Tables/Vtable.GFS Vtable #new line added; one time use

  rm -f "${INTER_FILE}":*
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
}

create_interfile() {
  local NC_DIR="$1"
  local INTER_FILE="$2"

  rm -f "${INTER_FILE}":*
  echo "--------------------------"
  echo " start creating interfile "
  echo "--------------------------"
  cd "$SCRIPT_DIR/python" || exit
  $PYTHON create_interfile.py -i "${NC_DIR}" -o "${WRF_MAINDIR}/WPS" "${FCST_YYYYMMDD}"
  cd "${WRF_MAINDIR}/WPS" || exit
  echo "---------------------------"
  echo " end of creating interfile "
  echo "---------------------------"
}

run_metgrid() {
  local METGRID_DIR="$1"

  rm -f "${METGRID_DIR}"/met_em.d0*
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
}

source "$SCRIPT_DIR/set_env_wrf.run.sh"

cd "${WRF_MAINDIR}/WPS" || exit

WPS_START_DATE="${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}:00:00"
WPS_END_DATE="${FCST_YY2}-${FCST_MM2}-${FCST_DD2}_${FCST_ZZ2}:00:00"

### GFS
mkdir -p "$NAMELIST_SUFF/gfs"
generate_namelist "namelist.wps_gfs_$NAMELIST_SUFF"

run_geogrid

rm -f GRIBFILE.*
# Update GFS link
rm -f "${WPS_GFSDIR}"/*.grb
ln -sf "${GFS_DIR}"/*.grb "${WPS_GFSDIR}"/.
run_ungrib "${WPS_GFSDIR}" "GFS"

run_metgrid "${NAMELIST_SUFF}/gfs"

rm -f geo_em*

### ECMWF
mkdir -p "$NAMELIST_SUFF/ecmwf"
generate_namelist "namelist.wps_ecmwf_$NAMELIST_SUFF"

run_geogrid

rm -f GRIBFILE.*
# Update ECMWF link
rm -f "$WPS_ECMWFDIR"/*.nc
ln -sf "$ECMWF_NC_DIR"/*.nc "$WPS_ECMWFDIR"/.

create_interfile "$WPS_ECMWFDIR" "ECMWF"

run_metgrid "${NAMELIST_SUFF}/ecmwf"

rm -f geo_em*
rm -f GFS:*
rm -f ECMWF:*
rm -f GRIB*
