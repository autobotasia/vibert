python3 create_pretraining_data.py \
  --input_file=./data/vi/v0.1/corpus-full-0.txt \
  --output_file=./output/tf_examples.tfrecord \
  --vocab_file=./vi-vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
