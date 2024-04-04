from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders, models

transcriptions = ""
dataset_path = "../dataset"

for stage in ["train", "val"]:
    with open(f'{dataset_path}/{stage}.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            print(line)
            transcriptions +=line.split("|")[1].strip()
            
with open("transcriptions.txt", "w",encoding='utf-8') as f:
  f.write(transcriptions.strip())
  
import re
import torch
from unidecode import unidecode
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def text_cleaners(text):
  ###########################################
  # ToDo Adjust this code for your language #
  ###########################################
  text = unidecode(text)
  text = text.lower()
  text = re.sub(re.compile(r'\s+'), ' ', text)
  text = text.replace('"', '')
  return text


def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(', '}': ')',
        '[': '(', ']': ')',
        '`': '\'', '—': '-',
        '—': '-', '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word

with open('transcriptions.txt', 'r', encoding='utf-8') as at:
    ttsd = at.readlines()
    allowed_characters_re = re.compile(r'^[a-zA-Z\u0900-\u094F!:;"/, \-\(\)\.\'\?\u0900-\u097F]+$')

    def preprocess_word(word, report=False):
        word = text_cleaners(word)
        word = remove_extraneous_punctuation(word)
        if not bool(allowed_characters_re.match(word)):
            if report and word:
                print(f"REPORTING: '{word}'")
            return ''
        return word

    def batch_iterator(batch_size=1000):
        print("Processing ASR texts.")
        for i in range(0, len(ttsd), batch_size):
            yield [preprocess_word(t, True) for t in ttsd[i:i+batch_size]]

    trainer = BpeTrainer(special_tokens=['[STOP]', '[UNK]', '[SPACE]'], vocab_size=255)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ttsd))
    tokenizer.save('custom_language_tokenizer.json')
  
