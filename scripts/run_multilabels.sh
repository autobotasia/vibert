BERT_BASE_DIR=cased_L-12_H-768_A-12-bert/cased_L-12_H-768_A-12/
DATA_DIR=data/multilabels/
python3 run_multilabels_classifier.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$DATA_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$DATA_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/intent_classification_base/
