# NanoGPT for English-to-Hindi Machine Translation
## Overview

NanoGPT is a lightweight custom GPT (Generative Pre-trained Transformer) model designed for the WMT2014 English-to-Hindi machine translation task. This model is implemented using PyTorch and is inspired by the Transformer architecture introduced in the paper `"Attention is All You Need"` by `Vaswani et al`.
[![gpt-vs-nano](https://i.postimg.cc/L5JRmC2Y/gpt-vs-nano-copy.jpg)

## Table of Contents
1. Introduction
2. Features
3. Installation
4. Usage
5. Model Architecture
6. Training
7. Inference
8. Acknowledgements

## Introduction
Machine translation is a challenging natural language processing task that involves translating text from one language to another. NanoGPT is specifically designed for the English-to-Hindi translation task, leveraging the power of the Transformer architecture to achieve state-of-the-art performance in a lightweight manner.

## Features
- Lightweight GPT model tailored for English-to-Hindi translation.
- Based on the Transformer architecture introduced in the paper "Attention is All You Need."
- Easy-to-use PyTorch implementation for training and inference.

## Installation
```bash
pip install torch
# Additional dependencies may be required. Refer to the requirements.txt file.
```

## Usage
To use NanoGPT for English-to-Hindi translation, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your_username/nanogpt.git
cd nanogpt
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Train the model using your dataset or use a pre-trained model.
4. Perform inference on new English sentences to get Hindi translations.

## Model Architecture
The NanoGPT model architecture is based on the Transformer introduced in the "Attention is All You Need" paper. It consists of multiple layers of self-attention and feedforward neural networks, enabling effective learning of contextual information for translation.

## Training
To train NanoGPT on the WMT2014 English-to-Hindi dataset, use the following command:

```bash
python train.py --dataset_path /path/to/wmt2014/dataset
```
Additional training options and hyperparameters can be configured in the `config.py` file.

## Inference
Performing inference with NanoGPT is straightforward. Simply load the pre-trained model and use it to translate English sentences to Hindi:

```python
from nanogpt import NanoGPT, translate_sentence

model = NanoGPT.load_model('/path/to/pretrained/model')
english_sentence = "Hello, how are you?"
hindi_translation = translate_sentence(model, english_sentence)
print(f"English: {english_sentence}\nHindi: {hindi_translation}")
```
Acknowledgements
This project is built upon the Transformer architecture and is heavily influenced by the works of `Vaswani et al.` in `"Attention is All You Need."`
## Author

- [@Anmol](https://github.com/anmol1512)


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://linktr.ee/AnmolChhetri2000)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anmol1512/)



## Acknowledgements

 - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v7.pdf)
