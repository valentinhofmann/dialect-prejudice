#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

for variable in groenwold blodgett race
do
    python3.10 -u ../probing/mgp_gpt3.py \
    --model davinci \
    --variable $variable \
    --attribute katz

    python3.10 -u ../probing/mgp_gpt3.py \
    --model text-davinci-003 \
    --variable $variable \
    --attribute katz
done
