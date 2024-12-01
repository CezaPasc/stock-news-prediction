from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb
from tqdm import tqdm
import os

import utils
from metrics import DirectionAcc, SentimentAcc, ProfitSimulation, AvgProfit

def find_eval_runs(baseline_model, dataset_name):
    # Check if baseline run already exists for this dataset:
    api = wandb.Api()
    project_path = "/".join(api.project(os.getenv("WANDB_PROJECT")).path)
    existing_runs = api.runs(
        path=project_path,
        filters={
            "config.model": baseline_model,
            "config.dataset": dataset_name
        }
    )
    
    return existing_runs


def eval_on_model(model_path, testset, dataset_name, eval_thresholds, scaler=None, random_state=None, group=None):
    model_name = model_path.split("/")[-1]
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": random_state,
    }
    
    run_name = "%s-%s" % (model_name, dataset_name)
    wandb.init(name=run_name, config=config, job_type="eval", group=group)
    
    # Load pre-trained FinBERT model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model in evaluation mode

    dataloader = DataLoader(testset, batch_size=32, shuffle=True)

    zero_scaled = eval_thresholds["zero_scaled"]
    #threshold_left, threshold_right = eval_thresholds["neutral_borders"]
    threshold_left, threshold_right = 1/3, 2/3 # Use thirds to have better comparison to guessing 

    metrics = [
        DirectionAcc(zero_scaled),
        SentimentAcc(threshold_left, threshold_right),
        ProfitSimulation(threshold=0.01, scaler=scaler),
        AvgProfit(threshold=0.01, scaler=scaler)
    ]

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        predictions = outputs.logits
        
        for m in metrics:
            m.update(predictions,labels)

    
    eval_metrics = {"eval/%s" % m.get_name(): m.compute().item() for m in metrics}

    # Logging the evaluated metrics for 50 epochs, to achieve them being displayed as a horizontal line on wandb.ai
    for e in range(50):
        eval_metrics["train/epoch"] = e
        wandb.log(eval_metrics, step=e)

    wandb.finish()
 


def eval_on_classification_model(model_name, testset, dataset_name, scaler=None, random_state=None, group=None):
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": random_state,
    }
    
    run_name = "%s_%s" % (model_name, dataset_name)
    wandb.init(name=run_name, config=config, group=group)
    
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set the model in evaluation mode

    dataloader = DataLoader(testset, batch_size=32, shuffle=True)

    zero_scaled = 0 if not scaler else scaler.transform([[0]])[0][0]
    threshold_left = -0.009 if not scaler else scaler.transform([[-0.009]])[0][0]
    threshold_right = 0.009 if not scaler else scaler.transform([[0.009]])[0][0]

    metrics = [
        DirectionAcc(zero_scaled),
        SentimentAcc(threshold_left, threshold_right),
        ProfitSimulation(threshold=0.009, scaler=scaler),
        AvgProfit(threshold=0.009, scaler=scaler)
    ]

    # TODO: Could be improved by scaling it between thresholds and maximum values using its confidence values (but not necessary for the current metrics)
    mmap = {
        0: 0.01 if not scaler else scaler.transform([[0.01]]),
        1: -0.01 if not scaler else scaler.transform([[-0.01]]),
        2: 0 if not scaler else scaler.transform([[0]]),
    }

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        mapped_predictions = torch.Tensor(np.array([mmap[p.item()] for p in predictions])).reshape(-1, 1)

        for m in metrics:
            m.update(mapped_predictions,labels)

    
    eval_metrics = {"eval/%s" % m.get_name(): m.compute().item() for m in metrics}

    # Logging the evaluated metrics for 50 epochs, to achieve them being displayed as a horizontal line on wandb.ai
    for e in range(50):
        eval_metrics["train/epoch"] = e
        wandb.log(eval_metrics, step=e)

    wandb.finish()
    

if __name__ == "__main__":
    # Example Usage
    from data import datasets
    
    os.environ["WANDB_PROJECT"]= "single_asset_training"
    model_name = "prosusai/finbert"
    text_column = "Title"
    target_column = "target_1h"
    scaling = None
    path = "test.csv"
    dataset_name = path.split("data/", 1)[1]
    
    df = utils.read_dataset(path, text_column, target_column=target_column)
    
    scaler = None
    if scaling:
        from sklearn.preprocessing import MinMaxScaler 
        scaler = MinMaxScaler()
        df[["target"]] = scaler.fit_transform(df[["target"]])

    train_data, test_data = datasets.train_test_split(df, attribute="ID", test_size=0.1)
    test = datasets.TextAugmentationDataset(test_data, model_name=model_name)
    
    eval_on_classification_model(model_name, test, target_column, scaler, group="FinBERT")
