# demo_2.py

# Model Name: Helsinki-NLP/opus-mt-en-hi
# !nvidia-smi
# !pip install datasets transformers[sentencepiece] sacrebleu -q
# !pip install pyarrow==14.0.1

import os
import sys
import transformers
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay

model_checkpoint = 'Helsinki-NLP/opus-mt-en-hi'
raw_datasets = load_dataset('cfilt/iitb-english-hindi')
raw_datasets

# Preprocessing the data
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 128
max_target_length = 128

source_lang = "en"
target_lang = "hi"

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    # Assign the tokenized labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
num_train_epochs = 1

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
train_dataset = model.prepare_tf_dataset(tokenized_datasets["test"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
validation_dataset = model.prepare_tf_dataset(tokenized_datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn=data_collator)

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)
model.fit(train_dataset, validation_data=validation_dataset, epochs=1)

model.save_pretrained("tf_model/")
