#!/bin/bash

DATA_ROOT=$HOME/data
DATA_DIR=${DATA_ROOT}/nyu2

if [ -d ${DATA_DIR} ];then
    echo "${DATA_DIR} already exists. Try again after removing it."
    echo "Aborted."
    exit 1
fi

mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=\
$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate\
 'https://docs.google.com/uc?export=download&id=1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw'\
 -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')\
&id=1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw" -O nyu2.zip
rm -f /tmp/cookies.txt

unzip nyu2.zip

rm -f nyu2.zip
