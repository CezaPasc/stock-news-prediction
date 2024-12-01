import wandb

import joblib
import os
import argparse

import utils
from data import datasets
import evaluation

def evaluate_run(run, dataset_dir, target_column="target", text_column=None):
    print(f"Evaluate {run.name} on test set...")
    # Get Metadata
    group = run.group
    model_name = run.config["_name_or_path"]
    eval_thresholds = run.config["thresholds"]
    if text_column is None:
        text_column = run.config["text_column"]
    
    # Load models and other artifacts
    run_artifacts = run.logged_artifacts()
    model = max([a for a in run_artifacts if a.type == "model"], key=lambda x: x.version)
    model_dir = model.download()
    
    # Load Scaler
    used_artifacts = run.used_artifacts()
    scaler = None
    scalers = [a for a in used_artifacts if a.type == "scaler"]
    if scalers:
        scaler_artifact = max(scalers, key=lambda x: x.version)
        scaler_dir = scaler_artifact.download()
        scaler = joblib.load(os.path.join(scaler_dir, "scaler.pkl"))
        
    # Load test set
    df = utils.read_dataset(os.path.join(dataset_dir, "test.csv"), text_column, target_column)
    if scaler:
        df[["target"]] = scaler.fit_transform(df[["target"]])

    test_set = datasets.TextAugmentationDataset(df, model_name=model_name) 
    
    # Run inference on test set
    evaluation.eval_on_model(model_dir, test_set, "testset", eval_thresholds, scaler, group=group)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Evaluate on Test Set',
                        description='Evaluate a group of experiments on a test set',
                        epilog='Text at the bottom of help')
    parser.add_argument('-d', '--dataset')      
    parser.add_argument('-e', '--experiment')      
    parser.add_argument('-i', '--input') # Is only required to add if the wandb config does not contain the right text column  
    parser.add_argument('-t', '--target')     
    parser.add_argument('-p', '--project')     

    project_name = "back-to-old"
    group_name = "target_1h"
    dataset_dir = None 
    target = "target_1h"
    text_column = None

    args = parser.parse_args()

    if args.input:
        text_column = args.input

    if args.target:
        target = args.target
        
    if args.dataset:
        dataset_dir = args.dataset
        
    if args.experiment:
        group_name = args.experiment

    if args.project:
        project_name = args.project
    
    os.environ["WANDB_PROJECT"] = project_name

    api = wandb.Api()
    filter = {
        "group": group_name,
        "jobType": {"$ne": "eval"} 
    }
    runs = api.runs(project_name, filters=filter)
    
    [evaluate_run(r, dataset_dir, target, text_column) for r in runs]
 