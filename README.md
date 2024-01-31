# wikiGPT

## Project Overview
wikiGPT is a comprehensive repository designed for training a GPT model with English Wikipedia data. It simplifies the process of downloading, preparing, and training GPT models, using Python scripts and Jupyter Notebooks. The project aims to demystify the training of Large Language Models (LLMs). The training code is simplified for readability and can be easily adapted for training on multiple GPUs. It incorporates recent PyTorch advancements like model compilation and mixed precision training.

## Installation

### For Linux and macOS:
```bash
git clone https://github.com/2p1987/wikiGPT.git
cd wikiGPT
python3 -m venv env
source env/bin/activate
pip install poetry
poetry install
```

### For Windows:
```bash
git clone https://github.com/2p1987/wikiGPT.git
cd wikiGPT
python -m venv env
.\env\Scripts\activate
pip install poetry
poetry install
```

## Usage
The workflow includes:
- `wikiGPT/prepare.py`: Loads and pre-tokenizes data for training.
- `wikiGPT/train.py`: Initiates model training.
- `wikiGPT/sample.py`: Generates text from a trained model.

Optional: `wikiGPT/tokenize.py` for training a new tokenizer.

### Prepare Data
#### Downloading and Saving Data Locally:
```bash
python -m wikiGPT.prepare shuffle
```
Outputs data shards in JSON format in `wikiGPT/data/shuffled_shards` from Wikipedia English. The share of Wikipedia can be modified in the script (50% by default).
Data source: https://huggingface.co/datasets/wikimedia/wikipedia

#### Pre-tokenizing Data for Training:
Pre-tokenize data to streamline training. For example:
```bash
python -m wikiGPT.prepare pretokenize --tokenizer_path wikiGPT/tokenizers/tok32000.model
```
Tokenizers are available in `wikiGPT/tokenizers/`. You can train your own with `wikiGPT/tokenize.py`.

### Train a Model
Refer to `wikiGPT/train.py` for training parameters, and `wikiGPT/model.py` for model parameters.

Example training command:
```bash
python -m wikiGPT.train [training parameters]
```

### Companion Notebooks/Scripts

The `wikiGPT/miscellaneous` folder contains 3 companion notebooks:
- `estimating_optimal_compute_parameters.ipynb`: Pre-computes the number of parameters of a given model and provides guidelines on the optimal volume of data to train it with.
- `count_total_tokens.py`: Counts the total number of tokens in the pre-tokenized data folder.
- `gpt_model_introduction.ipynb`: An introductory notebook aimed at providing information about how GPT models are trained to a non-specialized audience curious about understanding how LLM models work under the hood. (WIP)

## License
This project is under the MIT License - see LICENSE for details.

## Acknowledgments
- Forked from A. Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)
- The notebook estimating the [optimal compute parameters](wikiGPT/miscellaneous/estimating_optimal_compute_parameters.ipynb) is taken (and slightly adapted) from A. Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Model architecture based on Facebook's [Llama model](https://github.com/facebookresearch/llama)
