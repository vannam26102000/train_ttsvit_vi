# Vietnamese Speech Synthesis with VITS text to speech model and TTS Coqui framework

This repository is dedicated to the customization and training of VITS [(Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech)](https://arxiv.org/abs/2106.06103) for text-to-speech (TTS) applications using Vietnamese language data, utilizing the TTS Coqui framework.
The repository contains the necessary code and resources to train VITS specifically for generating high-quality speech from Vietnamese text. 


## Pre-requisites
1. I highly recommend you to use conda virtual environment, with Python 3.11.5.
```bash
conda create -n vits python=3.11.5
```
2. In this repo, I use TTS framework version 0.17.5 for statibility.
```bash
pip install TTS==0.17.5
```

## Inference
```python
from TTS.api import TTS

tts = TTS('vits_tts',
          model_path='path to the .pth file ',
          config_path='path to the config.json file')

tts.tts_to_file(text="Your example text", file_path="your_filename.wav")
```

## Demo
My trained model is published on this [HuggingFace space](/huggingface.co/spaces/Namkoy/train_vits_vi). Due to the resource factor to train the model, the results achieved are not as expected. The upcoming goal is to collect personal data for implementation voice clone.
