MODEL_PREFIX = "tokenizer" #@param {type: "string"}
VOC_SIZE = 32000 #@param {type:"integer"}
SUBSAMPLE_SIZE = 12800000 #@param {type:"integer"}
NUM_PLACEHOLDERS = 256 #@param {type:"integer"}

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
VOC_FNAME = "vocab.txt" #@param {type:"string"}

with open(VOC_FNAME, "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")
