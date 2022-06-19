#!/bin/bash

set -e

readonly PYTHON_CMD=python3
readonly LOG_FILE=run.sh.log
readonly TARGET_REWARD=$1

usage() {
    echo -e "" >&2
    echo -e "Usage: bash run.sh [target_reward]" >&2
    echo >&2
    echo "Arguments: " >&2
    echo -e "\ttarget_reward\tif reward >= target_reward, then exit" >&2

    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

if ! [[ "$TARGET_REWARD" =~ ^[0-9]+$ ]]; then
    echo -e "Err: target_reward must be integer\n"
    exit 1
fi

echo -n "" >$LOG_FILE

while true; do
    $PYTHON_CMD main.py
    out=$($PYTHON_CMD main.py --play --log)
    float_reward="${out//*[^-.0-9]/}"
    int_reward=${float_reward%.*}

    echo $out
    echo $out >>$LOG_FILE

    if [ $int_reward -ge $TARGET_REWARD ]; then
        break
    fi
done
