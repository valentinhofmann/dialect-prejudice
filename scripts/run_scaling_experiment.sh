#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

for variable in groenwold blodgett race
do
    for model in gpt2 gpt2-medium gpt2-large gpt2-xl roberta-base roberta-large t5-small t5-base t5-large t5-3b
    do
        python3.10 -u ../probing/mgp.py \
        --model $model \
        --variable $variable \
        --attribute katz \
        --device "$1"
    done

    python3.10 -u ../probing/mgp_gpt3.py \
    --model text-davinci-003 \
    --variable $variable \
    --attribute katz
done

for variable in groenwold blodgett
do
    for model in gpt2 gpt2-medium gpt2-large gpt2-xl
    do
        python3.10 -u ../perplexity/ppl_gpt2.py \
        --model $model \
        --variable $variable \
        --device "$1"
    done

    for model in roberta-base roberta-large
    do
        python3.10 -u ../perplexity/ppl_roberta.py \
        --model $model \
        --variable $variable \
        --device "$1"
    done

    for model in t5-small t5-base t5-large t5-3b
    do
        python3.10 -u ../perplexity/ppl_t5.py \
        --model $model \
        --variable $variable \
        --device "$1"
    done

    python3.10 -u ../perplexity/ppl_gpt3.py \
    --variable $variable
done
