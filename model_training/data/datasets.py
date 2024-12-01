from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split as split
import torch
import random

import config
import data.augmentation as augmentation

class TextAugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augmentation_type=None, augmentation_prob=0.1, model_name="bert-base-cased", max_length=512, token=None, verbose=False):
        self.dataset = dataset.reset_index(drop=True)
        self.verbose = verbose
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.huggingface_token)
        if token:
            self.tokenizer.pad_token_id = token
        self.max_length = max_length
        
        # Data Augmentation
        self.augmentation_type = augmentation_type
        self.augmentation_prob = augmentation_prob
        self.back_translator = augmentation.BackTranslation("en", "tl")
        #self.llm_augmenter = augmentation.LLMAugmenter()
        

    def __getitem__(self, index):
        instance = self.dataset.loc[index]
        text = instance['text']
        label = instance["target"]
        
        if self.verbose:
            print("Original Text: %s" % text)
            
        # Apply augmentation transformation
        text = self.augment_text(text, label)
        
        if self.verbose:        
            print("Augmented Text: %s" % text)

        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        target = torch.tensor(label, dtype=torch.float32)

        return {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target
        }

    def __len__(self):
        return len(self.dataset)
        
    def augment_text(self, text, score=None):
        if random.random() < self.augmentation_prob:
            return text
        
        # augmentation      
        match self.augmentation_type:
            case "SynonymReplacement":
                return augmentation.synonym_replacement(text, 2)
            case "RandomInsertion":
                return augmentation.random_insertion(text, 2)
            case "RandomSwap":
                return augmentation.random_swap(text, 2)
            case "RandomDeletion":
                return augmentation.random_deletion(text, 0.2)
            case "Noise":
                return augmentation.add_noise(text, 0.1)
            case "BackTranslation":
                return self.back_translator.augment(text)
            case "LLM":
                # return self.llm_augmenter.augment(text, score)
                raise NotImplementedError("The LLM Augmentation was removed out of the training pipeline, since it uses too much resources.")

        return text


def train_test_split(data, test_size=0.1, attribute=None, random_state=None):
    """
    Splits data into two non-overlapping subsets.
    
    Args:
        data (pandas.DataFrame): Data to be splitted.
        test_size (float): Represents the proportion of the dataset to include in the test split.
        attribute (str): If not None, there will be no overlapping instances having the same value for the specified attribute between the train and test set.
        random_state (int): Randomnization Seed.
        
    Returns:
        train_data (pandas.DataFrame)
        test_data (pandas.DataFrame)
    """
    if attribute is None:
        return split(data, test_size=test_size, random_state=random_state)
    unique_values = data[attribute].unique()
    train_values, test_values = split(unique_values, test_size=test_size, random_state=random_state)
     
    train_data = data[data[attribute].isin(train_values)]
    test_data = data[data[attribute].isin(test_values)]

    return train_data, test_data
