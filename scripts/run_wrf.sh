#!/bin/bash
source "$SCRIPT_DIR/set_env_wrf.run.sh"

# -------------------------------------------- #
#     Link the ICBC in this model directory    #
# -------------------------------------------- #
cd "$WRF_REALDIR" || exit
rm -f met_em.d0*

NUM_TIMESTEPS=$((WRF_FCST_DAYS * 24 + 1))
MET_EM_FILES=("$WRF_MAINDIR/WPS/$WPS_NAMELIST_SUFF"/met_em.d0*)
if [ ${#MET_EM_FILES[@]} -ne $NUM_TIMESTEPS ]; then
	err_msg="number of met_em files: ${#MET_EM_FILES[@]}, expected: $NUM_TIMESTEPS"
	echo "$err_msg" >>"$ERROR_FILE"
	echo "$err_msg"
	exit 1
fi

ln -s "$WRF_MAINDIR/WPS/$WPS_NAMELIST_SUFF"/met_em.d0* .
for f in met_em.*; do mv -v "$f" "$(echo "$f" | tr ':' '_')"; done

# -------------------------------------------- #
#             Edit namelist.input              #
# -------------------------------------------- #

rm -f namelist.input
sed -i "2s/.*/\ run_days                            = ${WRF_FCST_DAYS},/" "namelist.input_${NAMELIST_RUN}"
sed -i "6s/.*/\ start_year                          = ${FCST_YY},${FCST_YY},/" "namelist.input_${NAMELIST_RUN}"
sed -i "7s/.*/\ start_month                         = ${FCST_MM},${FCST_MM},/" "namelist.input_${NAMELIST_RUN}"
sed -i "8s/.*/\ start_day                           = ${FCST_DD},${FCST_DD},/" "namelist.input_${NAMELIST_RUN}"
sed -i "9s/.*/\ start_hour                          = ${FCST_ZZ},${FCST_ZZ},/" "namelist.input_${NAMELIST_RUN}"
sed -i "12s/.*/\ end_year                            = ${FCST_YY2},${FCST_YY2},/" "namelist.input_${NAMELIST_RUN}"
sed -i "13s/.*/\ end_month                           = ${FCST_MM2},${FCST_MM2},/" "namelist.input_${NAMELIST_RUN}"
sed -i "14s/.*/\ end_day                             = ${FCST_DD2},${FCST_DD2},/" "namelist.input_${NAMELIST_RUN}"
sed -i "15s/.*/\ end_hour                            = ${FCST_ZZ2},${FCST_ZZ2},/" "namelist.input_${NAMELIST_RUN}"
ln -s "namelist.input_${NAMELIST_RUN}" namelist.input

# -------------------------------------------- #
#                Run Real.exe                  #
# -------------------------------------------- #
echo "********************"
echo "*   start of real  *"
echo "********************"
touch rsl.error.0000
rm -f wrfbdy* wrfinput*
srun ./real.exe >&log.real &
tail --pid=$! -f rsl.error.0000
if ! tail -n 1 "rsl.error.0000" | grep -q "SUCCESS"; then
  echo "real" >>"$ERROR_FILE"
  rm -f rsl.error.* rsl.out.*
  echo "********************"
  echo "*    real error    *"
  echo "********************"
  exit 1
fi
rm -f rsl.error.* rsl.out.*
echo "********************"
echo "*    end of real   *"
echo "********************"

if [[ $WRF_MODE == '3dvar' ]]; then
	mkdir -p "$WRF_REALDIR/wrfreal_tmp"
	mv wrfbdy_d01 "$WRF_REALDIR/wrfreal_tmp/wrfbdy_d01_${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}"
	mv wrfinput_d01 "$WRF_REALDIR/wrfreal_tmp/wrfinput_d01_${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}"

	source "$SCRIPT_DIR/run_wrfda_pre.sh"

	cd "$WRF_REALDIR" || exit
	ln -s "$WRF_MAINDIR/WRF3DVar/wrfinput_d01" .
	ln -s "$WRF_MAINDIR/WRF3DVar/wrfbdy_d01" .
fi

# -------------------------------------------- #
#                Run WRF.exe                   #
# -------------------------------------------- #
echo "********************"
echo "*   start of wrf   *"
echo "********************"
touch rsl.error.0000
srun ./wrf.exe >&log.wrf &
tail --pid=$! -f rsl.error.0000
if ! tail -n 1 "rsl.error.0000" | grep -q "SUCCESS"; then
  echo "wrf" >>"$ERROR_FILE"
  rm -f rsl.error.* rsl.out.*
  echo "********************"
  echo "*     wrf error    *"
  echo "********************"
  exit 1
fi
rm -f rsl.error.* rsl.out.*
echo "********************"
echo "*    end of wrf    *"
echo "********************"

mkdir -p "$NAMELIST_RUN"

mv "wrfout_d01_${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}_00_00" "$NAMELIST_RUN/wrfout_d01_${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}_00_00_${WRF_FCST_DAYS}-day_fcst_rain"

if [[ $WRF_MODE == '3dvar' ]]; then
	rm "$WRF_REALDIR/wrfreal_tmp/wrfinput_d01_${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}"
	rm "$WRF_REALDIR/wrfreal_tmp/wrfbdy_d01_${FCST_YY}-${FCST_MM}-${FCST_DD}_${FCST_ZZ}"
fi
