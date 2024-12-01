import pandas as pd
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import config
from data import stats, utils, sampling

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Data Split',
                        description='Splits and samples the data',
                        epilog='Text at the bottom of help')
    parser.add_argument('-d', '--dataset')      
    parser.add_argument('-o', '--output')      
    parser.add_argument('-t', '--target')      

    
    text_column = "Title" # This is "Title" for the news and "text" for social media
    target = "target_1h"
    test_size=0.1
    generate_augmented_dataset = True
    # Can be set to true, if the test set should be created independently from the target variable. This can be helpful when evaluating different target variable options on the same dataset.
    targetIndependentTestSet = False 

    publishers = [
        "reuters",
        "businesswire",
        "prnewswire",
        "globenewswire",
        "bloomberg",
        "finance",
        "ap",
        "fortune",
    ]
    
    args = parser.parse_args()
    
    path = None
    path = os.path.join(config.data_dir, path)
    if args.dataset:
        path = args.dataset
    
    if args.target:
        target = args.target
    
    # Read Dataset
    df = pd.read_csv(path)

    if args.output:
        dirname = args.output
        os.makedirs(dirname, exist_ok=True)
    else:
        dirname = path.replace(".csv", "")
        os.makedirs(dirname, exist_ok=True)
        filename = path.split("/")[-1]
        df.to_csv(os.path.join(dirname, filename))
        
    
    dirname = os.path.join(dirname, target)
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(os.path.join(dirname, "res"), exist_ok=True)

    # Remove NaN Values
    print("Rows before dropping nan:", len(df))
    df.dropna(subset=[target], axis=0, inplace=True)
    print("Rows after dropping nan:", len(df))
    
    # Show histogram
    if "Publisher" in df:
        stats.plot_publisher_freq(df, "01_publisher_freq", path=dirname)
    stats.plot_dist(df[target], "02_hist", path=dirname)

    # Filter to selected publishers
    if len(publishers) > 0 and "Publisher" in df:
        df = df[df["publisher"].isin(publishers)]
        
        stats.plot_publisher_freq(df, "01_publisher_freq_selection", path=dirname)
        stats.plot_dist(df[target], "02_hist_publisher_selection", path=dirname)

    if targetIndependentTestSet:
        df, test_set = train_test_split(df, test_size=test_size, random_state=42)
    
    # Filter out outliers
    outliers = utils.find_outliers(df[target], threshold=1)
    filtered = df.loc[~outliers]
    print("Rows after filtering outliers:", len(filtered))
    stats.plot_dist(filtered[target], "03_non_outliers_hist", path=dirname)

    scaler = MinMaxScaler()
    filtered[["target_scaled"]] = scaler.fit_transform(filtered[[target]])
    filtered["augmented"] = False
    
    if not targetIndependentTestSet:
        # Divide into train and test set
        train_indices, test_indices = [], []
        # Sample into n groups by sentiment
        groups = sampling.get_sampled_groups(filtered, sample_size=-1, min_len=1, column=target)
        for group in groups.values():
            # Add groups containing only one element directly to train, since they can not be splitted
            if len(group) == 1:
                train_indices += group.index.to_list()
                continue
            
            # Draw 10% for test set
            train, test = train_test_split(group.index, test_size=test_size, random_state=42)
            train_indices += train.to_list()
            test_indices += test.to_list()

        train_set = filtered.loc[train_indices]
        test_set = filtered.loc[test_indices]
    else:
        # Test Set was already created in previous step
        train_set = filtered
        
    train_set.to_csv(os.path.join(dirname, "train_raw.csv"))
    # Laying test set aside
    test_set.to_csv(os.path.join(dirname, "test.csv"))

    # Sample by Borders
    borders = [-0.01, 0, 0.01]
    sample_borders = sampling.get_samples_by_borders(train_set, borders, -1, 1, target)
    sampled_borders = sampling.get_samples_by_smallest_group(sample_borders.values(), min_len=100)
    stats.plot_dist(sampled_borders[target], "06_sampled_borders_hist", path=dirname)
    sampled_borders.to_csv(os.path.join(dirname, "train_border_sampled.csv"))

    # Sample by groups
    sample_groups = sampling.get_sampled_groups(train_set, 10, -1, 1, target)
    sampled = sampling.get_samples_by_smallest_group(sample_groups.values(), min_len=100)
    stats.plot_dist(sampled[target], "04_sampled_hist", path=dirname)
    sampled.to_csv(os.path.join(dirname, "train_group_sampled.csv"))

    # Simple Random Sampling
    random_sample_size = max([len(sampled_borders), len(sampled)])
    random_sampled = train_set.sample(n=random_sample_size)
    stats.plot_dist(random_sampled[target], "05_random_sampled_hist", path=dirname)
    sampled.to_csv(os.path.join(dirname, "train_random_sampled.csv"))

    # Sample using augmentation
    if generate_augmented_dataset:
        sampled = sampling.get_samples_by_augmenting(sample_groups.values(), desired_size=300, text_column=text_column)
        stats.plot_dist(sampled[target], "06_augmented_sampled_hist", path=dirname)
        sampled.to_csv(os.path.join(dirname, "train_augmented_sampled.csv"))
