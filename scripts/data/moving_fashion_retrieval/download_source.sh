#!/usr/bin/env bash
set -euo pipefail

# Download the official MovingFashion archive with byte-range resume support.
#
# Usage:
#   ./scripts/data/moving_fashion_retrieval/download_source.sh [OUTPUT] [PARTIAL]
#
# OUTPUT defaults to ./movingfashion.zip. PARTIAL defaults to OUTPUT.part. To
# continue a browser download, pass its .crdownload path as PARTIAL after the
# browser has stopped writing to it. The partial file is preserved on errors
# and atomically renamed to OUTPUT only after its exact size is verified.

readonly SOURCE_URL="https://bit.ly/4bTZGeS"
readonly LAST_KNOWN_SHARE_URL="https://gofile-334e684f37.fr1.quickconnect.to/sharing/qZYHiHJpf"
readonly DOWNLOAD_FILENAME="movingfashion.zip"
readonly EXPECTED_BYTES="25925189869"
readonly MAX_ATTEMPTS="50"

output_path="${1:-movingfashion.zip}"
partial_path="${2:-${output_path}.part}"

if [[ "${output_path}" == "${partial_path}" ]]; then
    echo "OUTPUT and PARTIAL must be different paths." >&2
    exit 2
fi

mkdir -p "$(dirname "${output_path}")" "$(dirname "${partial_path}")"

cookie_jar="$(mktemp "${TMPDIR:-/tmp}/movingfashion-cookies.XXXXXX")"
cleanup() {
    rm -f "${cookie_jar}"
}
trap cleanup EXIT INT TERM

if [[ -f "${output_path}" ]]; then
    output_bytes="$(wc -c < "${output_path}" | tr -d ' ')"
    if [[ "${output_bytes}" == "${EXPECTED_BYTES}" ]]; then
        echo "MovingFashion archive is already complete: ${output_path}"
        exit 0
    fi
    echo "Existing OUTPUT has an unexpected size: ${output_bytes} bytes" >&2
    echo "Move it aside or pass a different OUTPUT path." >&2
    exit 2
fi

if [[ -f "${partial_path}" ]]; then
    before_bytes="$(wc -c < "${partial_path}" | tr -d ' ')"
    sleep 2
    after_bytes="$(wc -c < "${partial_path}" | tr -d ' ')"
    if [[ "${before_bytes}" != "${after_bytes}" ]]; then
        echo "PARTIAL is still being written by another downloader:" >&2
        echo "  ${partial_path}" >&2
        echo "Pause or stop the browser download before running this script." >&2
        exit 2
    fi
    if (( after_bytes > EXPECTED_BYTES )); then
        echo "PARTIAL is larger than the official archive." >&2
        exit 2
    fi
    echo "Resuming at ${after_bytes} of ${EXPECTED_BYTES} bytes."
else
    echo "Starting a new MovingFashion download."
fi

# Follow the official source URL on every run because Synology may change the
# QuickConnect relay hostname. Some gofile.me responses perform their final
# redirect in JavaScript, which curl cannot execute; retain the verified relay
# as a fallback in that case.
share_url="$(curl \
    --fail \
    --location \
    --connect-timeout 30 \
    --retry 5 \
    --retry-all-errors \
    --output /dev/null \
    --write-out '%{url_effective}' \
    "${SOURCE_URL}")"
share_url="${share_url%%\?*}"
share_url="${share_url%/}"
if [[ "${share_url}" != */sharing/* ]]; then
    echo "Official redirect uses JavaScript; using the verified Synology relay."
    share_url="${LAST_KNOWN_SHARE_URL}"
fi
relay_base="${share_url%%/sharing/*}"
sharing_id="${share_url##*/sharing/}"
download_url="${relay_base}/fsdownload/${sharing_id}/${DOWNLOAD_FILENAME}"

# Synology binds the anonymous sharing cookie to the connection route. Visit the
# share and start the file transfer as two operations in one curl process so its
# connection can be reused. `--continue-at -` derives the Range start from the
# existing PARTIAL size and leaves those bytes intact if the network drops.
# Restart the whole two-request sequence after a failure so each retry gets a
# fresh cookie and recomputes the resume offset from the current file size.
attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
    echo "Download attempt ${attempt}/${MAX_ATTEMPTS}."
    if curl \
        --fail \
        --location \
        --connect-timeout 30 \
        --retry 5 \
        --retry-delay 5 \
        --retry-all-errors \
        --cookie-jar "${cookie_jar}" \
        --cookie "${cookie_jar}" \
        --output /dev/null \
        "${share_url}" \
        --next \
        --fail \
        --location \
        --connect-timeout 30 \
        --retry 0 \
        --speed-limit 1024 \
        --speed-time 120 \
        --cookie-jar "${cookie_jar}" \
        --cookie "${cookie_jar}" \
        --continue-at - \
        --output "${partial_path}" \
        "${download_url}"
    then
        break
    else
        curl_status="$?"
        current_bytes="$(wc -c < "${partial_path}" | tr -d ' ')"
        if [[ "${current_bytes}" == "${EXPECTED_BYTES}" ]]; then
            break
        fi
        if (( attempt == MAX_ATTEMPTS )); then
            echo "Download failed after ${MAX_ATTEMPTS} attempts." >&2
            exit "${curl_status}"
        fi
        echo "Transfer interrupted at ${current_bytes}/${EXPECTED_BYTES} bytes; reconnecting in 10 seconds." >&2
        sleep 10
        ((attempt += 1))
    fi
done

actual_bytes="$(wc -c < "${partial_path}" | tr -d ' ')"
if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" ]]; then
    echo "Download remains incomplete: ${actual_bytes}/${EXPECTED_BYTES} bytes." >&2
    echo "Rerun this script to continue." >&2
    exit 1
fi

mv "${partial_path}" "${output_path}"
echo "MovingFashion download complete: ${output_path}"
