#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

for variable in g_dropping aint habitual been stay copula inflection finna
do
    for model in gpt2 gpt2-medium gpt2-large gpt2-xl roberta-base roberta-large t5-small t5-base t5-large t5-3b
    do
        python3.10 -u ../probing/mgp.py \
        --model $model \
        --variable $variable \
        --attribute katz \
        --device "$1"
    done
done
