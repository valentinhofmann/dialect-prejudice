![](https://drive.google.com/uc?id=1NvBNuPNFH3FHEOe4ImIXp4aFK6DmbfNR)

# Dialect Prejudice in Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python: 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
<a target="_blank" href="https://colab.research.google.com/github/valentinhofmann/dialect-prejudice/blob/main/demo/matched_guise_probing_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>   


## Overview

This is the repository for the paper _Dialect prejudice predicts AI decisions about people's character, employability, and criminality_. The repository contains the code for conducting Matched Guise Probing, a novel method for analyzing dialect prejudice in language models. Furthermore, the repository contains a demo illustrating how to use the code as well as scripts and notebooks for replicating the experiments and analyses from the paper.


## Setup

All requirements can be found in `requirements.txt`. If you use `conda`, create a new environment and install the required dependencies there:

```
conda create -n dialect-prejudice python=3.10
conda activate dialect-prejudice
pip install -r requirements.txt
```

Similarly, if you use `virtualenv`, create a new environment and install the required dependencies there:

```
python -m virtualenv -p python3.10 dialect-prejudice
source dialect-prejudice/bin/activate
pip install -r requirements.txt
```

The setup should only take a few moments.

## Usage

Matched Guise Probing requires three types of data: two sets of texts that differ by dialect (e.g., African American English and Standard American English), a set of tokens that we want to analyze (e.g., trait adjectives), and a set of prompts. Put the two sets of texts as a tab-separated text file into `data/pairs`.
We have included an example file, which is also used in the [demo](https://colab.research.google.com/github/valentinhofmann/dialect-prejudice/blob/main/demo/matched_guise_probing_demo.ipynb). Put the set of tokens 
as a text file into `data/attributes`. `data/attributes` contains several example files (e.g., the trait adjectives from the Princeton Trilogy used in the paper). Finally, define the set of prompts in `probing/prompting.py`. `probing/prompting.py` contains all prompts used in the paper.

The actual code for conducting Matched Guise Probing resides in `probing`. 



matched - unmatched

stores predictions in probs -> notebooks show how these predictions can then be loaded

decisions

## Demo 

We have created a [demo](https://colab.research.google.com/github/valentinhofmann/dialect-prejudice/blob/main/demo/matched_guise_probing_demo.ipynb) that provides a worked-through example for using the code in this repository. Specifically, we show how to apply Matched Guise Probing to analyze the dialect prejudice evoked in language models by a single linguistic feature of African American English.

## Repruduction

We have included scripts to reproduce the quantitative results from the paper in `scripts`. The scripts expect the data from [Blodgett et al. (2016)](https://slanglab.cs.umass.edu/TwitterAAE/) and [Groenwold et al. (2020)](https://aclanthology.org/2020.emnlp-main.473/) as tab-separated text files in `data/pairs`. To replicate all experiments, run:

```
bash scripts/run_stereotype_experiment.sh $device
bash scripts/run_feature_experiment.sh $device
bash scripts/run_employability_experiment.sh $device
bash scripts/run_criminality_experiment.sh $device
bash scripts/run_scaling_experiment.sh $device
bash scripts/run_human_feedback_experiment.sh $device
```

Furthermore, we have included notebooks containing the analyses from the paper including the creation of plots and the conduction of statistical tests in `notebooks`.