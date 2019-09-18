#!/bin/bash
#set -eu

export PYTHONPATH="$(pwd)/:$PYTHONPATH"
source activate ai3

cpt_dir=./checkpoint/$(date "+%d_%b_%Y_%H_%M_%S")

echo "$0 $@"

mkdir -p $cpt_dir

# by Kaldi NISR_SRE_2018 M=8
python -u ./train_speech_embedder.py > $cpt_dir/train.log
