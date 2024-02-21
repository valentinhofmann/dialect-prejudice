# Dialect Prejudice in Language Models

<a target="_blank" href="https://colab.research.google.com/github/valentinhofmann/dialect-prejudice/blob/main/demo/matched_guise_probing_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Overview

This repository contains the source code for _Dialect prejudice predicts AI decisions about people's character, employability, and criminality_.

![](https://drive.google.com/uc?id=1NvBNuPNFH3FHEOe4ImIXp4aFK6DmbfNR)


## Setup

All requirements can be found in `requirements.txt`. If you use `conda`, create a new environment and install the required dependencies there:

```bash
conda create -n dialect-prejudice python=3.10
conda activate dialect-prejudice
pip install -r requirements.txt
```

If you use `virtualenv`, you can do:

```bash
python -m virtualenv -p python3.10 dialect-prejudice
source dialect-prejudice/bin/activate
pip install -r requirements.txt
```

Setup should only take a few minutes.
