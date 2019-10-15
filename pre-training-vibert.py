import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
import sentencepiece as spm

from glob import glob
from tensorflow.keras.utils import Progbar
from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder
  
# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s :  %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
log.handlers = [sh]

LANG_CODE = "vi"

DEMO_MODE = True
if DEMO_MODE:
 CORPUS_SIZE = 1000000
else:
 CORPUS_SIZE = 100000000

regex_tokenizer = nltk.RegexpTokenizer("\w+")

def normalize_text(text):
  # lowercase text
  text = str(text).lower()
  # remove non-UTF
  text = text.encode("utf-8", "ignore").decode()
  # remove punktuation symbols
  text = " ".join(regex_tokenizer.tokenize(text))
  return text

def count_lines(filename):
  count = 0
  with open(filename) as fi:
    for line in fi:
      count += 1
  return count 

RAW_DATA_FPATH = "data/vi/binhvq/v0.1/corpus-full-10.txt"
PRC_DATA_FPATH = "proc_dataset.txt"

# apply normalization to the dataset
# this will take a minute or two

total_lines = count_lines(RAW_DATA_FPATH)
bar = Progbar(total_lines)

with open(RAW_DATA_FPATH,encoding="utf-8") as fi:
  with open(PRC_DATA_FPATH, "w",encoding="utf-8") as fo:
    for l in fi:
      fo.write(normalize_text(l)+"\n")
      bar.add(1)


MODEL_PREFIX = "tokenizer"
VOC_SIZE = 32000
SUBSAMPLE_SIZE = 12800000
NUM_PLACEHOLDERS = 256

SPM_COMMAND = ('--input={} --model_prefix={} '
               '--vocab_size={} --input_sentence_size={} '
               '--shuffle_input_sentence=true ' 
               '--bos_id=-1 --eos_id=-1').format(
               PRC_DATA_FPATH, MODEL_PREFIX, 
               VOC_SIZE - NUM_PLACEHOLDERS, SUBSAMPLE_SIZE)

spm.SentencePieceTrainer.Train(SPM_COMMAND)

def read_sentencepiece_vocab(filepath):
  voc = []
  with open(filepath, encoding='utf-8') as fi:
    for line in fi:
      voc.append(line.split("\t")[0])
  # skip the first <unk> token
  voc = voc[1:]
  return voc

snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))
print("Learnt vocab size: {}".format(len(snt_vocab)))
print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))


def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token
        
bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))

ctrl_symbols = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
bert_vocab = ctrl_symbols + bert_vocab

bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
print(len(bert_vocab))


VOC_FNAME = "vi-vocab.txt"

with open(VOC_FNAME, "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")

