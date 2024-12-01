# ## Log in to Wandb in terminal
# `wandb login`

import wandb

import torch
from sklearn.preprocessing import MinMaxScaler 
from transformers import TrainingArguments, AutoModelForSequenceClassification, set_seed

import sys
import os 
import random
import argparse
import joblib

from data import datasets
from metrics import DirectionAcc, SentimentAcc, ProfitSimulation, AvgProfit
import utils
import evaluation
import config

# save your trained model checkpoint to wandb
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
# Save trained model at the end
os.environ["WANDB_LOG_MODEL"] = "end"

project_name = "single-asset-training"
scaling = True
text_column = "Title"
target_column = "target_1h"
augmentation_type = "BackTranslation"
augmentation_prob = 0.9
path = None
base_model = "bert-base-cased"
eval_on_baseline = True
baseline_model = "prosusai/finbert"
fixed_seed = True 
random_state = 42 if fixed_seed else random.randint(1,1000)
sentiment_threshold = None #(-0.02, 0.02)


parser = argparse.ArgumentParser(
                    prog='Fine Tune',
                    description='Fine Tune a pre-trained Model',
                    epilog='Text at the bottom of help')
parser.add_argument('-d', '--dataset')      
parser.add_argument('-e', '--experiment')      
parser.add_argument('-i', '--input')      
parser.add_argument('-t', '--target')      
parser.add_argument('-p', '--project')     
parser.add_argument('-n', '--runs')     

args = parser.parse_args()
if args.input:
    text_column = args.input
    
if args.target:
    target_column = args.target
    print("use target:", target_column)

if args.project:
    project_name = args.project

os.environ["WANDB_PROJECT"] = project_name

if args.dataset:
    path = args.dataset
    
dataset_name = path.split("data/", 1)[1]

df = utils.read_dataset(path, text_column, target_column=target_column)

df["target_raw"] = df["target"]

scaler = None
if scaling:
    scaler = MinMaxScaler()
    df[["target"]] = scaler.fit_transform(df[["target"]])
    # Prepare scaler for being logged
    joblib.dump(scaler, "scaler.pkl")
    scaler_artifact = wandb.Artifact("scaler", type="scaler", description="Scaler for the output variable")
    scaler_artifact.add_file("scaler.pkl")
    with wandb.init():
        wandb.log_artifact(scaler_artifact)


zero_scaled = 0 if not scaling else scaler.transform([[0]])[0][0]

if sentiment_threshold:
    lower, upper = sentiment_threshold
    threshold_left = lower if not scaling else scaler.transform([[lower]])[0][0]
    threshold_right = upper if not scaling else scaler.transform([[upper]])[0][0]
else:
    # Use same threshold values as the previous run
    threshold_left = 1/3
    threshold_right = 2/3


train_data, val_data = datasets.train_test_split(df, attribute="ID", test_size=0.1, random_state=random_state)

train_set =  datasets.TextAugmentationDataset(train_data, augmentation_type=augmentation_type, model_name=base_model, augmentation_prob=augmentation_prob)
val_set = datasets.TextAugmentationDataset(val_data, model_name=base_model) 


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


def train(seed=None, group=None):
    # Clear GPU memory before starting training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model_seed = seed if seed else random.randint(1,1000)
    set_seed(model_seed)
    
    # Models
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1, token=config.huggingface_token) # 1 Label since its a regression task
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Accelartion for Silicon Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model.to(device)

    training_args = TrainingArguments(
        output_dir='models',
        report_to="wandb",
        logging_steps=5,
        num_train_epochs=6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        eval_steps=20,
        save_steps = 100,
        neftune_noise_alpha=5,
        learning_rate=0.0002445,
        warmup_ratio=0.2,
        weight_decay=0.4,
    )

    from trainer import CustomTrainer

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
    )

    cfg = {
        "scaling": scaling,
        "scaler": type(scaler),
        "text_column": text_column,
        "dataset": dataset_name,
        "seed": random_state,
        "model_seed": model_seed,
        "augmentation_type": train_set.augmentation_type,
        "thresholds": {
            "zero_scaled": zero_scaled,
            "neutral_borders": [threshold_left, threshold_right]
        }
    }

    with wandb.init(config=cfg, group=group) as run:
        trainer.train()
        # Save scaler
        if scaler:
            run.use_artifact(scaler_artifact)


if __name__ == "__main__":
    if eval_on_baseline:
        existing_runs = evaluation.find_eval_runs(baseline_model, dataset_name)
        if len(existing_runs) == 0:
            print("Starting baseline evaluation of model `%s` for dataset `%s`" % (baseline_model, dataset_name))
            evaluation.eval_on_classification_model(baseline_model, val_set, dataset_name, scaler, random_state)
        else:
            print("Baseline evaluation of model `%s` on dataset `%s` exists already" % (baseline_model, dataset_name))

    group_name = None
    if args.experiment:
        group_name = args.experiment

    group_name = group_name if group_name else wandb.util.generate_id()

    # Run multiple training runs to eliminate uncertainties due to randomness 
    num_runs = 3
    if args.runs:
        num_runs = int(args.runs)
        
    seeds = random.sample(range(1,1000), num_runs)
    for seed in seeds:
        train(seed=seed, group=group_name)
