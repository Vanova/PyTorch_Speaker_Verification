#!/bin/bash
#set -eu

export PYTHONPATH="$(pwd)/:$PYTHONPATH"
source activate ai3

#cpt_dir=./checkpoint/$(date "+%d_%b_%Y_%H_%M_%S")

echo "$0 $@"

#mkdir -p $cpt_dir

python -u dvector_create_sre.py --datadir /home/vano/wrkdir/projects_data/sre_2019/swbd_sre_small_fbank/ > dvec_swbd_sre_small_fbank.log
