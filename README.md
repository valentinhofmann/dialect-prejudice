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
git clone https://github.com/valentinhofmann/dialect-prejudice.git
cd dialect-prejudice
pip install -r requirements.txt
```

Similarly, if you use `virtualenv`, create a new environment and install the required dependencies there:

```
python -m virtualenv -p python3.10 dialect-prejudice
source dialect-prejudice/bin/activate
git clone https://github.com/valentinhofmann/dialect-prejudice.git
cd dialect-prejudice
pip install -r requirements.txt
```

The setup should only take a few moments.

## Usage

Matched Guise Probing requires three types of data: two sets of texts that differ by dialect (e.g., African American English and Standard American English), a set of tokens that we want to analyze (e.g., trait adjectives), and a set of prompts. Put the two sets of texts as a tab-separated text file into `data/pairs`.
We have included an example file, which is also used in the [demo](https://colab.research.google.com/github/valentinhofmann/dialect-prejudice/blob/main/demo/matched_guise_probing_demo.ipynb). Put the set of tokens 
as a text file into `data/attributes`. `data/attributes` contains several example files (e.g., the trait adjectives from the Princeton Trilogy used in the paper). Finally, define the set of prompts in `probing/prompting.py`. `probing/prompting.py` contains all prompts used in the paper.

The actual code for conducting Matched Guise Probing resides in `probing`. Simply run the following command:

```
python3.10 mgp.py \
--model $model \
--variable $variable \
--attribute $attribute \
--device $device
```

The meaning of the individual arguments is as follow:

- `$model` is the name of the model being used (e.g., `t5-large`).
- `$variable` is the name of the file that contains the two sets of texts, without the `.txt` extension.
- `$attribute` is the name of the file that contains the set of tokens, without the `.txt` extension.
- `$device` specifies the device on which to run the code.


For OpenAI models, you need to put your OpenAI API key into a file called `.env` at the root of the repository (e.g., `OPENAI_KEY=123456789`). We also use separate Python files to conduct Matched Guise Probing with OpenAI models. For example, you can run the following command for GPT4:

```
python3.10 mgp_gpt4.py \
--model $model \
--variable $variable \
--attribute $attribute
```

To run experiments that ask the models to make a discrete decision for each input text (e.g., the conviction experiment in the paper), you can use the same syntax as for general Matched Guise Probing. Simply put the decision tokens as a text file into `data/attributes` and specify a set of suitable prompts in `probing/prompting.py`. Since the models might assign different prior probabilities to the decision tokens, we recommend to use calibration based on the token probabilities in a neutral context. To do so, you can use the `--calibrate` argument.

All prediction probabilities are stored in `probing/probs`. We have included examples in `notebooks` that show how to load and analyze these prediction probabilities. Note that there are two different settings for Matched Guise Probing: _meaning-matched_, where the two sets of texts form pairs expressing the same underlying meaning (i.e., the two tab-separated texts on each line in the text file belong together), and _non-meaning-matched_, where the two sets of texts are independent from each other. The file `notebooks/helpers.py` contains two functions for loading predictions in these two settings (i.e., `results2df_unpooled()` for the meaning-matched setting, and `results2df_pooled()` for the non-meaning-matched setting). Alternatively, you can also add the name of the text file to the lists `UNPOOLED_VARIABLES` or `POOLED_VARIABLES` in `notebooks/helpers.py` and use the function `results2df()`.



## Demo 

We have created a [demo](https://colab.research.google.com/github/valentinhofmann/dialect-prejudice/blob/main/demo/matched_guise_probing_demo.ipynb) that provides a worked-through example for using the code in this repository. Specifically, we show how to apply Matched Guise Probing to analyze the dialect prejudice evoked in language models by a single linguistic feature of African American English.

## Reproduction

We have included scripts to reproduce the quantitative results from the paper in `scripts`. The scripts expect the data from [Blodgett et al. (2016)](https://slanglab.cs.umass.edu/TwitterAAE/) and [Groenwold et al. (2020)](https://aclanthology.org/2020.emnlp-main.473/) as tab-separated text files in `data/pairs` (see [above](#usage)). To replicate all experiments, run:

```
bash scripts/run_stereotype_experiment.sh $device
bash scripts/run_feature_experiment.sh $device
bash scripts/run_employability_experiment.sh $device
bash scripts/run_criminality_experiment.sh $device
bash scripts/run_scaling_experiment.sh $device
bash scripts/run_human_feedback_experiment.sh $device
```

Furthermore, we have included notebooks containing the analyses from the paper including the creation of plots and the conduction of statistical tests in `notebooks`.