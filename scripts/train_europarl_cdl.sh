#!/bin/bash
output_dir='checkpoints/seq2seq-europarl-sc/ebnodb_5_15_ep2_Polar5G_cdlA_fulltest'
trainset_path='data/europarl/processed/train.csv'
devset_path='data/europarl/processed/test.csv'

mkdir -p $output_dir

# --channel_type AWGN \
# --ebno_db 10 \

python train.py \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --num_train_epochs 2 \
    --do_train \
    --do_eval \
    --model_name_or_path facebook/bart-base \
    --preprocessing_num_workers 4 \
    --save_total_limit 1 \
    --no_use_fast_tokenizer \
    --num_beams 1 \
    --max_source_length 64 \
    --max_target_length 64 \
    --train_file "$trainset_path" \
    --validation_file "$devset_path" \
    --test_file "$devset_path" \
    --output_dir $output_dir \
    --ebno_db_min 5 \
    --ebno_db_max 15 \
    --channel_type CDL \
    --fec_type Polar5G \
    --cdl_model "A" \
    --channel_num_tx_ant 2 \
    --channel_num_rx_ant 2 \
    --num_bits_per_symbol 4 \
    --overwrite_output_dir \
    --tokenizer_name facebook/bart-base \
    --pad_to_max_length \
    --dataset_config 3.0.0 \
    --bin_conv_method tanh