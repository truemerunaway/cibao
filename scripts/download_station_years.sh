#!/usr/bin/env bash
set -u

BASE="${BASE:-/root/autodl-tmp/data/intermagnet_raw}"
START_YEAR="${START_YEAR:-2008}"
END_YEAR="${END_YEAR:-2022}"
PUBLICATION_STATE="${PUBLICATION_STATE:-definitive}"
ORIENTATION="${ORIENTATION:-XYZF}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:17890}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 STATION [STATION ...]"
  echo "Example: PROXY_URL=http://127.0.0.1:17890 $0 ABG"
  exit 2
fi

if [ -n "$PROXY_URL" ]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
fi

is_leap_year() {
  local year="$1"
  if [ $((year % 400)) -eq 0 ] || { [ $((year % 4)) -eq 0 ] && [ $((year % 100)) -ne 0 ]; }; then
    return 0
  fi
  return 1
}

expected_rows() {
  local year="$1"
  if is_leap_year "$year"; then
    echo 527040
  else
    echo 525600
  fi
}

download_year() {
  local station="$1"
  local year="$2"
  local next=$((year + 1))
  local outdir="${BASE}/${station}/${year}"
  local out="${outdir}/${station}_${year}_${PUBLICATION_STATE}_minute_${ORIENTATION}.txt"
  local tmp="${out}.tmp"
  local log="${outdir}/download_${station}_${year}.log"
  local expected
  local count

  mkdir -p "$outdir"
  expected="$(expected_rows "$year")"

  echo "===== Download ${station} ${year} ====="
  echo "Output: $out"

  wget \
    --tries=20 \
    --waitretry=10 \
    --read-timeout=300 \
    --timeout=120 \
    --no-http-keep-alive \
    -O "$tmp" \
    "https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&format=iaga2002&observatoryIagaCode=${station}&samplesPerDay=Minute&dataStartDate=${year}-01-01&dataEndDate=${next}-01-01&publicationState=${PUBLICATION_STATE}&orientation=${ORIENTATION}&recordTermination=UNIX" \
    > "$log" 2>&1

  count="$(grep -c "^${year}-" "$tmp" || true)"

  if grep -qi "<html\|error\|exception" "$tmp"; then
    echo "BAD: ${station} ${year} downloaded error page, keep old file" | tee -a "$log"
    return 1
  fi

  if [ "$count" -eq "$expected" ]; then
    mv "$tmp" "$out"
    echo "OK: ${station} ${year} count=${count}, replaced ${out}" | tee -a "$log"
    return 0
  fi

  echo "BAD: ${station} ${year} count=${count}, expected=${expected}, keep old file" | tee -a "$log"
  return 1
}

for station in "$@"; do
  mkdir -p "${BASE}/${station}"
  summary="${BASE}/${station}/download_${station}_${START_YEAR}_${END_YEAR}.log"
  {
    echo "Station: ${station}"
    echo "Years: ${START_YEAR}-${END_YEAR}"
    echo "Base: ${BASE}"
    echo "Publication state: ${PUBLICATION_STATE}"
    echo "Orientation: ${ORIENTATION}"
    echo "Proxy: ${PROXY_URL:-none}"
    echo
  } > "$summary"

  for year in $(seq "$START_YEAR" "$END_YEAR"); do
    if download_year "$station" "$year"; then
      echo "${year} OK" >> "$summary"
    else
      echo "${year} BAD" >> "$summary"
    fi
  done
done
