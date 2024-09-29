# Translation Projects

This repository contains three different projects for translating text between various languages using machine learning models.

## Demos

### 1. English to German Translation
- File: `demo_1.py`
- Uses Hugging Face Transformers for translating English text to German using a pipeline.

### 2. English to Hindi Translation
- File: `demo_2.py`
- Uses the Helsinki-NLP model (`Helsinki-NLP/opus-mt-en-hi`) to translate English text to Hindi. This demo includes preprocessing steps and training functionalities.

### 3. Multi-Language Translation with Evaluation
- File: `demo_3.py`
- Uses the M2M100 model for translating between multiple languages and evaluates the translations using BLEU and ROUGE scores.

## Prerequisites

Before running any of the demos, make sure to install the required libraries:
```bash
pip install torch torchvision torchaudio
pip install transformers ipywidgets gradio --upgrade
pip install datasets transformers[sentencepiece] sacrebleu -q
pip install pyarrow==14.0.1
pip install evaluate gradio requests rouge_score
