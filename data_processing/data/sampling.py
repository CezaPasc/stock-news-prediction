import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import sys

from data.augmentation import LLMAugmenter


def get_samples_by_smallest_group(groups, min_len) -> pd.DataFrame:
    samples = []

    smallest_group = min([len(g) for g in groups])
    sample_len = max(min_len, smallest_group)

    for group in groups:
        if sample_len > len(group):
            sample_len = len(group)
        samples.append(group.sample(n=sample_len, replace=False))

    return pd.concat(samples)
    
def get_samples_by_augmenting(groups, desired_size, factor=9, text_column="text") -> pd.DataFrame:
    llm_augmenter = LLMAugmenter(factor=factor)
    samples = []
    for group in tqdm(groups):
        available_size = min(desired_size, len(group))
        remaining_size = max(0, desired_size-available_size)
        samples.append(group.sample(n=available_size, replace=False))

        print(f"Available: {available_size}, Remaining: {remaining_size}")
        
        if remaining_size:
            print("We should augment here")
            amount_to_augment = int(np.ceil(remaining_size/factor))
            amount_to_augment = min(available_size, amount_to_augment)
            samples_to_augment = group.sample(n=amount_to_augment, replace=False)
            
            augmented_instances = llm_augmenter.augment_batch(samples_to_augment, text_column=text_column)
            amount_augmented = min(len(augmented_instances), remaining_size)
            samples.append(augmented_instances.sample(n=amount_augmented))
    
    return pd.concat(samples)

def get_samples_by_borders(data, borders, sample_size=200, min_len=1, column="target", oversampling=False):
    '''
    This method takes a pandas DataFrame of data and 
    returns a dictionary of sampled groups based on the values in a specified column.

    Parameters: 
        data (DataFrame): A pandas DataFrame.
        borders (list): A list of border values defining the boundaries of each stratum.
        sample_size (int): The number of samples to draw from each group. If -1, then all samples are returned. Default is 200.
        min_len (int): Minimum amount of observations of a group to consider it to be added.
        column (str): The name of the column to use for grouping. Default is "target".
        oversampling (bool): Allow oversampling. Default is False.

    Returns:
    samples (dict): A dictionary of DataFrames, where the keys are the group names and the values are the sampled DataFrames.
    '''
    conditions = []
    choices = []

    for i in range(-2, len(borders) - 1):
        lower = borders[i + 1] if i != -2 else -np.inf
        upper = borders[i + 2] if i != len(borders) - 2 else np.inf

        condition = (data[column] > lower) & (data[column] <= upper)
        conditions.append(condition)
        choices.append(i + 2)

    levels = np.select(conditions, choices)
    data["levels"] = levels

    grouped = data.groupby("levels")
    grouped_sizes = grouped.size()
    print(grouped_sizes)
    data.drop(["levels"], inplace=True, axis=1)
    dfs = {k: x for k, x in grouped}

    samples = {}
    for key in dfs.keys():
        if sample_size == -1:
            samples[key] = dfs[key]
            continue

        if len(dfs[key]) < min_len:
            continue

        replace = False if oversampling else len(dfs[key]) < sample_size
        samples[key] = dfs[key].sample(n=sample_size, replace=replace)

    return samples

def get_sampled_groups(data, n_groups=10, sample_size=200, min_len=15, column="target", oversampling=False):
    '''
    This method takes a pandas DataFrame of data and 
    returns a dictionary of sampled groups based on the values in a specified column.

    Parameters: 
        data (DataFrame): A pandas DataFrame of data with numeric values between 0 and 1.
        n_groups (int): The number of groups to create based on the column values. Default is 10
        sample_size (int): The number of samples to draw from each group. If -1, then all samples are returned. Default is 200.
        min_len (int): Minimum amount of observations of a group to consider it to be added.
        column (str): The name of the column to use for grouping. Default is "target".
        oversampling (bool): Allow oversampling. Default is False.

    Returns:
    samples (dict): A dictionary of DataFrames, where the keys are the group names and the values are the sampled DataFrames.
    '''
    conditions = []
    choices = []

    lowest = min(data[column]) - sys.float_info.epsilon
    maximum = max(data[column]) + sys.float_info.epsilon

    for i in range(n_groups):
        lower = lowest + (i / n_groups) * (maximum - lowest)
        upper = lowest + ((i + 1) / n_groups) * (maximum - lowest)
        condition = (data[column] > lower) & (data[column] <= upper)
        conditions.append(condition)
        choices.append("%d.  %f - %f" % (i, lower, upper))

    levels = np.select(conditions, choices)
    data["levels"] = levels

    grouped = data.groupby("levels")
    grouped_sizes = grouped.size()
    print(grouped_sizes)
    data.drop(["levels"], inplace=True, axis=1)
    dfs = {k: x for k, x in grouped}

    samples = {}
    for key in dfs.keys():
        if sample_size == -1:
            samples[key] = dfs[key]
            continue

        if len(dfs[key]) < min_len:
            continue

        replace = len(dfs[key]) < sample_size and oversampling
        if not replace and sample_size > len(dfs[key]):
            sample_size = len(dfs[key])
        
        samples[key] = dfs[key].sample(n=sample_size, replace=replace)

    return samples
