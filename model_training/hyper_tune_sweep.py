# ## Log in to Wandb in terminal
# `wandb login`

import wandb

import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler 

import evaluate
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification, set_seed

from matplotlib import pyplot as plt
import sys
import os 
import random

from data import datasets
from metrics import DirectionAcc, SentimentAcc, ProfitSimulation, AvgProfit
import utils
import evaluation
import config
from trainer import CustomTrainer


# save your trained model checkpoint to wandb
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


scaling = True 
text_column = "Title"
augmentation_type = None 
path = "<PATH_TO_TRAIN_SET>"
base_model = "bert-base-cased"
eval_on_basseline = True
baseline_model = "prosusai/finbert"
fixed_seed = True 
random_state = 42 if fixed_seed else random.randint(1,1000)

if len(sys.argv) == 2:
    path = (sys.argv[1])
dataset_name = path.split("data/", 1)[1]


df = utils.read_dataset(path, text_column)

df["target_raw"] = df["target"]

scaler = None
if scaling:
    scaler = MinMaxScaler()
    df[["target"]] = scaler.fit_transform(df[["target"]])

zero_scaled = 0 if not scaling else scaler.transform([[0]])[0][0]
threshold_left = -0.01 if not scaling else scaler.transform([[-0.01]])[0][0]
threshold_right = 0.01 if not scaling else scaler.transform([[0.01]])[0][0]

# Models
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1, token=config.huggingface_token) # 1 Label since its a regression task

train_data, test_data = datasets.train_test_split(df, attribute="ID", test_size=0.1, random_state=random_state)

train_set =  datasets.TextAugmentationDataset(train_data, augmentation_type=augmentation_type, model_name=base_model, augmentation_prob=0.1, token=model.config.eos_token_id)
test_set = datasets.TextAugmentationDataset(test_data, model_name=base_model, token=model.config.eos_token_id) 


metrics = [
    DirectionAcc(zero_scaled),
    SentimentAcc(threshold_left, threshold_right),
    ProfitSimulation(threshold=0.01, scaler=scaler),
    AvgProfit(threshold=0.01, scaler=scaler)
]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu()
        labels = labels.cpu()

    for m in metrics:
        m.update(logits, labels)
        
    results = {m.get_name(): m.compute().item() for m in metrics} 
    [m.reset() for m in metrics]

    return results


def train(cfg=None):
    global model
    del model
    # Clear GPU memory before starting training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_seed = random.randint(1,1000)
    set_seed(model_seed)
    
    with wandb.init(config=cfg):
        cfg = wandb.config
        cfg.seed = model_seed

        # Create Training Arguments using cfg
        training_args = TrainingArguments(
            output_dir='models',
            logging_steps=5,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            eval_steps=20,
            save_steps = 100,
            neftune_noise_alpha=cfg.neftune_noise_alpha,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
        )
        
        # Set Augmentation Method of Training Pipeline
        train_set.augmentation_type = cfg.train_augmentation
        train_set.augmentation_prob = cfg.train_augmentation_prob

        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1, token=config.huggingface_token) # 1 Label since its a regression task
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Acceleration for Silicon Macs
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        model.to(device)

        # Clear GPU memory before starting training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize Trainer using cfg
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=test_set,
            compute_metrics=compute_metrics,
        )

        # Start training run
        trainer.train()


if __name__ == "__main__":
    search_space = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'eval/profit_simulation',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 0.0003
            },
            'per_device_train_batch_size': {
                'values': [8, 16, 32]
            },
            'num_train_epochs': {
                'values': [3, 5, 6, 8, 10, 12]
            },
            'neftune_noise_alpha': {
                'values': [None, 10, 20]
            },
            'train_augmentation': {
                'values': [None, "SynonymReplacement", "RandomInsertion", "RandomSwap", "RandomDeletion", "Noise", "BackTranslation"]
            },
            'train_augmentation_prob': {
                'values': [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
            },
            'weight_decay': {
                'values': [0.2, 0.3, 0.4, 0.5]
            },
            'warmup_ratio': {
                'values': [0.15, 0.175, 0.2, 0.225, 0.25]  
            },
        }
    }

    sweep_id = wandb.sweep(search_space, project="furtherHP")
    wandb.agent(sweep_id, train)
