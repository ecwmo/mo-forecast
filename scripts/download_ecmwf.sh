#!/usr/bin/env bash

echo "---------------------------------"
echo " Downloading latest ECMWF files... "
echo "---------------------------------"

# Set variables
DL_LIST=$TEMP_DIR/ecmwf_dl_list.txt
DL_OUT=$TEMP_DIR/ecmwf_dl.out
DL_LOG_DIR=input/ecmwf_files
DL_SPEED_LOG_FILE="${DL_LOG_DIR}/ecmwf_dl_speed_${FCST_YY}${FCST_MM}.log"

mkdir -p "$ECMWF_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p "$DL_LOG_DIR"

END_TS=$((WRF_FCST_DAYS * 24)) # Last timestep to download

# URL
BASE_URL="https://data.ecmwf.int/forecasts/"
FILE_PARAM="${FCST_YYYYMMDD}${FCST_ZZ}0000-"
DIR_PARAM="${FCST_YYYYMMDD}/${FCST_ZZ}z/ifs/0p25/oper/"

# Create download list
touch "$DL_LIST"
for i in $(seq -f "%03g" 0 3 $END_TS); do
  if [ ! -f "$ECMWF_DIR/ECMWF$i.grib2" ]; then
    FCST_HH=$((10#$i)) 
    URL="${BASE_URL}${DIR_PARAM}${FILE_PARAM}$(printf "%d" $FCST_HH)h-oper-fc.grib2"
    echo "${URL}" >>"$DL_LIST"
    echo -e "\tout=ECMWF$i.grib2" >>"$DL_LIST"
  fi
done

# Wait for files to be available
TEST_URL="${BASE_URL}${DIR_PARAM}${FILE_PARAM}0h-oper-fc.grib2"
i=0
while [ "$i" -lt 10 ] && ! (curl -o/dev/null -sfI "$TEST_URL"); do
  sleep 10m
  i=$((i + 1))
  echo "waiting for ecmwf files to be available"
done

# Exit if file still doesn't exist
if ! (curl -o/dev/null -sfI "$TEST_URL"); then
  exit 1
fi

DATE_STR=$(date +"%Y%m%d,%H")
# Record start time
DL_STIME=$(date +%s)


# Download
DL_LOG="${DL_LOG_DIR}/ecmwf_dload_${FCST_YYYYMMDD}${FCST_ZZ}.log"
aria2c -j5 -x8 -d "$ECMWF_DIR" -i "$DL_LIST" -l "$DL_LOG" --log-level=notice

# Record end time
DL_ETIME=$(date +%s)
# Compute elapsed time in seconds
DL_DURATION=$((DL_ETIME - DL_STIME))
# Get total download size
TOT_SIZE=$(du -sk "$ECMWF_DIR"/ | cut -f1)
# Compute download speed
DL_SPEED=$(bc <<<"scale=2; $TOT_SIZE/$DL_DURATION")
# Log Speed
echo "$DATE_STR,$DL_DURATION,$TOT_SIZE,$DL_SPEED" >>"$DL_SPEED_LOG_FILE"

# Cleanup
rm -f "$DL_LIST"
rm -f "$DL_OUT"

echo "---------------------------------"
echo " Finished downloading ecmwf files! "
echo "---------------------------------"

# Convert each .grib2 file to .nc using cdo
for GRIB2_FILE in "$ECMWF_DIR"/*.grib2; do
  NC_FILE="${GRIB2_FILE%.grib2}.nc"
  echo "$GRIB2_FILE"
  cdo -f nc copy "$GRIB2_FILE" "$ECMWF_NC_DIR/$NC_FILE"
done

cd "$MAINDIR" || exit