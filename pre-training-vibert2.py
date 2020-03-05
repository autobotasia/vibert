
MAX_SEQ_LENGTH = 128 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param
MAX_PREDICTIONS = 20 #@param {type:"integer"}
DO_LOWER_CASE = True #@param {type:"boolean"}

PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}
# controls how many parallel processes xargs can create
PROCESSES = 2 #@param {type:"integer"}


XARGS_CMD = ("ls ./shards/ | "
             "xargs -n 1 -P {} -I{} "
             "python3 bert/create_pretraining_data.py "
             "--input_file=./shards/{} "
             "--output_file={}/{}.tfrecord "
             "--vocab_file={} "
             "--do_lower_case={} "
             "--max_predictions_per_seq={} "
             "--max_seq_length={} "
             "--masked_lm_prob={} "
             "--random_seed=34 "
             "--dupe_factor=5")

XARGS_CMD = XARGS_CMD.format(PROCESSES, '{}', '{}', PRETRAINING_DIR, '{}', 
                             VOC_FNAME, DO_LOWER_CASE, 
                             MAX_PREDICTIONS, MAX_SEQ_LENGTH, MASKED_LM_PROB)
                             
tf.gfile.MkDir(PRETRAINING_DIR)


BUCKET_NAME = "bert_resourses" #@param {type:"string"}
MODEL_DIR = "bert_model" #@param {type:"string"}
tf.gfile.MkDir(MODEL_DIR)

if not BUCKET_NAME:
  log.warning("WARNING: BUCKET_NAME is not set. "
              "You will not be able to train the model.")

# use this for BERT-base

bert_base_config = {
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": VOC_SIZE
}

with open("{}/bert_config.json".format(MODEL_DIR), "w") as fo:
  json.dump(bert_base_config, fo, indent=2)
  
with open("{}/{}".format(MODEL_DIR, VOC_FNAME), "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")


BUCKET_NAME = "bert_resourses" #@param {type:"string"}
MODEL_DIR = "bert_model" #@param {type:"string"}
PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}
VOC_FNAME = "vocab.txt" #@param {type:"string"}

# Input data pipeline config
TRAIN_BATCH_SIZE = 128 #@param {type:"integer"}
MAX_PREDICTIONS = 20 #@param {type:"integer"}
MAX_SEQ_LENGTH = 128 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param

# Training procedure config
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
TRAIN_STEPS = 1000000 #@param {type:"integer"}
SAVE_CHECKPOINTS_STEPS = 2500 #@param {type:"integer"}
NUM_TPU_CORES = 8

if BUCKET_NAME:
  BUCKET_PATH = "gs://{}".format(BUCKET_NAME)
else:
  BUCKET_PATH = "."

BERT_GCS_DIR = "{}/{}".format(BUCKET_PATH, MODEL_DIR)
DATA_GCS_DIR = "{}/{}".format(BUCKET_PATH, PRETRAINING_DIR)

VOCAB_FILE = os.path.join(BERT_GCS_DIR, VOC_FNAME)
CONFIG_FILE = os.path.join(BERT_GCS_DIR, "bert_config.json")

INIT_CHECKPOINT = tf.train.latest_checkpoint(BERT_GCS_DIR)

bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
input_files = tf.gfile.Glob(os.path.join(DATA_GCS_DIR,'*tfrecord'))

log.info("Using checkpoint: {}".format(INIT_CHECKPOINT))
log.info("Using {} data shards".format(len(input_files)))  

model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=INIT_CHECKPOINT,
      learning_rate=LEARNING_RATE,
      num_train_steps=TRAIN_STEPS,
      num_warmup_steps=10,
      use_tpu=USE_TPU,
      use_one_hot_embeddings=True)

tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)

run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=BERT_GCS_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=SAVE_CHECKPOINTS_STEPS,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)
  
train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=MAX_SEQ_LENGTH,
        max_predictions_per_seq=MAX_PREDICTIONS,
        is_training=True)

estimator.train(input_fn=train_input_fn, max_steps=TRAIN_STEPS)        