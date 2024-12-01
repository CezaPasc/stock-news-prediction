import random
import os
import re

from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer
from transformers import pipeline
from nltk.corpus import wordnet
import nltk
import torch
import numpy as np

import config


nltk.download('wordnet')

# This map is being used for augmentations on the character level
_keyboard = np.array(
    [
       ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
       ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']'],
       ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", ''],
       ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', '', ''],
    ]
)

class HuggingfaceChatTemplate:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.huggingface_token)
        self.tokenizer.use_default_system_prompt = False

    def get_chat(self, text: str, label: float) -> str:
        system_prompt = "You are an advanced AI writer. Your job is to help write examples of text with a certain sentiment. The examples should have the aim to differ from the input text. Sentiment is represented as a score between 0 and 1, where 0 means it has a very negative effect on the stock market price and 1 means it has a very positive effect on the stock market price."
        task = f"Based on the following news article title which has a {label} sentiment score, write 9 new similar examples in style of a news title, that has the same sentiment. Separate the texts by newline.",
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nText: {text}\nAnswer: """.format(
                    task=task,
                    text=text,
                ),
            },
        ]

        return chat


class LLMAugmenter:
    """
    Inspired from https://github.com/AndersGiovanni/worker_vs_gpt/blob/main/src/worker_vs_gpt/prompt_augmentation_hf.py
    """
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0.3) -> None:
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print("Creating pipeline")
        self.pipe = pipeline("text-generation",
                             model_name,
                             token=config.huggingface_token,
                             device=self.device,
                             torch_dtype=torch.float16
        )
        print("Pipeline created")

    def get_chat(self, text: str, label: float) -> str:
        system_prompt = "You are an advanced AI writer. Your job is to help write examples of text with a certain sentiment. The examples should have the aim to differ from the input text. Sentiment is represented as a score between 0 and 1, where 0 means it has a very negative effect on the stock market price and 1 means it has a very positive effect on the stock market price."
        task = f"Based on the following news article title which has a {label} sentiment score, write 9 new similar examples in style of a news title, that has the same sentiment. Separate the texts by newline.",
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": """{task}\nText: {text}\nAnswer: """.format(
                    task=task,
                    text=text,
                ),
            },
        ]


        return chat
        
    def augment(self, text, score):
        output = self.pipe(self.get_chat(text, score))
        """
        output = self.llm.text_generation(
            self.template.format(
                text=text,
                label=score,
            ),
            max_new_tokens=2048,
            temperature=self.temperature,
            repetition_penalty=1.2,
            return_full_text=False,
            truncate=4096,
        )
        """
        pattern = r'^\d+\.\s+(.+)'
        examples = re.findall(pattern, output, re.MULTILINE)

        return random.choice(examples)

class BackTranslation:
    def __init__(self, source_language, target_language, verbose=False):
        self.source_language = source_language
        self.target_language = target_language
        
        self.verbose = verbose
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        forward_name = "Helsinki-NLP/opus-mt-%s-%s" % (source_language, target_language)
        self.forward =  {
            "model": MarianMTModel.from_pretrained(forward_name).to(self.device),
            "tokenizer": MarianTokenizer.from_pretrained(forward_name)
        }
        
        back_name = "Helsinki-NLP/opus-mt-%s-%s" % (target_language, source_language)
        self.back =  {
            "model": MarianMTModel.from_pretrained(back_name).to(self.device),
            "tokenizer": MarianTokenizer.from_pretrained(back_name)
        }

    def augment(self, text):
        if self.verbose:
            print("Input:", text)
            
        # Translate the input text to the target language
        input_ids = self.forward["tokenizer"].encode(text, return_tensors="pt").to(self.device)
        output_ids = self.forward["model"].generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
        translated_text = self.forward["tokenizer"].decode(output_ids[0], skip_special_tokens=True)

        if self.verbose:
            print("Translated:", translated_text)

        # Translate the translated text back to the source language
        input_ids = self.back["tokenizer"].encode(translated_text, return_tensors="pt").to(self.device)
        output_ids = self.back["model"].generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
        back_translated_text = self.back["tokenizer"].decode(output_ids[0], skip_special_tokens=True)

        return back_translated_text


def _get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(text, n):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if _get_synonyms(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = _get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def random_insertion(text, n):
    words = text.split()
    for _ in range(n):
        new_word = _get_synonyms(random.choice(words))
        if new_word:
            words.insert(random.randint(0, len(words)), random.choice(new_word))
    return ' '.join(words)

def random_swap(text, n):
    words = text.split()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def random_deletion(text, p):
    words = text.split()
    if len(words) == 1:
        return text
    # TODO: Improve remaining
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

def _letter_swap(char_array):
    # Swap a random letter with its neighbor
    swap_idx = random.randint(1, len(char_array)-1)
    char_array[swap_idx-1], char_array[swap_idx] = char_array[swap_idx], char_array[swap_idx-1]
    
    return ''.join(char_array)
   
    
def _typo(char_array):
    # Replaces letter by nearby characters on the keyboard
    key_pos = ([],[])
    while not len(key_pos[0]):
        idx = random.randint(0, len(char_array)-1)
        selected_char = char_array[idx]
        key_pos = np.where(_keyboard == selected_char.upper())
    
    # Determining nearby key
    key_pos_array = np.array(key_pos).T
    typo_vector = np.random.choice([-1, 0, 1], size=key_pos_array.shape)
    typo_pos = key_pos_array + typo_vector
    
    # Clip and get char of typo in keyboard array
    typo_pos = np.clip(typo_pos, 0, np.array(_keyboard.shape) - 1)
    typo_pos = tuple([i for i in typo_pos[0]])
    typo = _keyboard[typo_pos]
    
    # Restore case of original char
    typo = typo.lower() if selected_char.islower() else typo
    
    char_array[idx] = typo
    
    return ''.join(char_array)

def add_noise(text, p):
    words = text.split()
    words = [w for w in words if len(w) > 1]
    aug_methods = [_letter_swap, _typo]
    for i in range(len(words)):
        # Pick random word
        if random.random() < p:
            char_array = list(words[i])
            aug_method = random.choice(aug_methods)
            words[i] = aug_method(char_array)
            
    return ' '.join(words)


if __name__ == "__main__":
    # Example usage
    text = "The stock market showed significant growth last quarter."

    augmented_text = add_noise(text, 0.2)
    print("Character Level:", augmented_text)

    augmented_text = synonym_replacement(text, 2)
    print("Synonym Replacement:", augmented_text)

    augmented_text = random_insertion(text, 2)
    print("Random Insertion:", augmented_text)

    augmented_text = random_swap(text, 2)
    print("Random Swap:", augmented_text)

    augmented_text = random_deletion(text, 0.2)
    print("Random Deletion:", augmented_text)
