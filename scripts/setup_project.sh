#!/bin/bash
# clone repository
git clone https://github.com/2p1987/climateGPT.git
cd climateGPT
python -m venv climate-gpt

# install project env
source climate-gpt/bin/activate
pip install poetry
poetry install

#downlaod data
python -m climateGPT.prepare shuffle
