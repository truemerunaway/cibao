#!/usr/bin/env bash
set -u

BASE="${BASE:-/root/autodl-tmp/data/intermagnet_raw}"
START_YEAR="${START_YEAR:-2008}"
END_YEAR="${END_YEAR:-2022}"
PUBLICATION_STATE="${PUBLICATION_STATE:-definitive}"
ORIENTATION="${ORIENTATION:-XYZF}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 STATION [STATION ...]"
  echo "Example: $0 FUR ABG"
  exit 2
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

for station in "$@"; do
  echo "===== ${station} ====="
  for year in $(seq "$START_YEAR" "$END_YEAR"); do
    file="${BASE}/${station}/${year}/${station}_${year}_${PUBLICATION_STATE}_minute_${ORIENTATION}.txt"
    expected="$(expected_rows "$year")"

    if [ ! -f "$file" ]; then
      echo "${year} MISSING expected=${expected}"
      continue
    fi

    count="$(grep -c "^${year}-" "$file" || true)"
    last="$(grep "^${year}-" "$file" | tail -1 | awk '{print $1" "$2}')"

    if grep -qi "<html\|error\|exception" "$file"; then
      echo "${year} BAD error_page count=${count} expected=${expected}"
    elif [ "$count" -eq "$expected" ]; then
      echo "${year} OK count=${count} last=${last}"
    else
      echo "${year} BAD count=${count} expected=${expected} last=${last}"
    fi
  done
done
