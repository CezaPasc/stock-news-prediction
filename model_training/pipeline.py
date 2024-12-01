import wandb

import os
import subprocess

def run_script_and_wait(script_path, log_file, *args):
    """
    Runs a Python script and waits until it completes.
    
    Args:
        script_path (str): The path to the Python script to run.
        *args: Additional arguments to pass to the script.
    
    Returns:
        int: The return code of the script (0 indicates success).
    """
    try:
        # Construct the command
        command = ["conda", "run", "-n", "training", "python", script_path] + list(args)
        
        # Run the script and wait until it finishes
        result = subprocess.run(command, capture_output=True, text=True)
        # Open the log file
        with open(log_file, "w") as log:
            # Run the script and wait until it finishes
            result = subprocess.run(command, stdout=log, stderr=log, text=True)
         
        # Return the exit code
        return result.returncode
    except FileNotFoundError:
        print(f"Error: Script '{script_path}' not found.")
        return -1


def fetch_accuracies_by_group_and_job(project_name, group_name, job_type):
    api = wandb.Api()
    runs = api.runs(project_name)
    
    return [r.summary.get("eval/accuracy") for r in runs if r.group == group_name and r.job_type == job_type]


if __name__ == "__main__":
    group_name = "return_12h"
    target_name = "target_12h"
    
    # Set new project name: 
    project = "test-run"
    output_path = f"../data/yahoo/{project}"
    dataset_path = os.path.join(output_path, target_name)
    
    # Generate Augmented Dataset
    print("Generating Dataset... (see data.logs)")
    data_split_script = "../data_processing/data_split.py"
    run_script_and_wait(data_split_script, "data.logs", "-o", output_path)

    # Fine-Tune Model with this Data 3 Times
    # Use settings of the fourth try
    print("Running Fine-Tunings... (see tune.logs)")
    augmented_set = os.path.join(dataset_path, "train_augmented_sampled.csv")
    run_script_and_wait("fine_tune.py", "tune.logs", "-d", augmented_set, "-e", group_name, "-t", target_name, "-p", project)

    # Run evaluation on it
    print("Running Evals... (see eval.logs)")
    run_script_and_wait("eval_on_test.py", "eval.logs", "-d", dataset_path, "-e", group_name, "-t", target_name, "-p", project)

    # Check results
    print("Checking results...")
    accs = fetch_accuracies_by_group_and_job(project, group_name, "eval")

    min_acc = min(accs)
    
    print("Best Lowest Acc:", min_acc)


