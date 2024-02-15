#!/bin/bash
output_dir='checkpoints/seq2seq-europarl-sc/ebnodb_3_10_ep3_awgn_jit_compiletest'
trainset_path='data/europarl/processed/train.csv'
devset_path='data/europarl/processed/test.csv'

mkdir -p $output_dir

# --model_name_or_path facebook/bart-base \
# --model_name_or_path '/home/danny911kr/joohan/seq2seq-sc2/checkpoints/seq2seq-europarl-sc/ebnodb_8_num_beams_1_ep30_nembeddings1024' \
# --model_name_or_path '/home/danny911kr/joohan/seq2seq-sc2/checkpoints/seq2seq-europarl-sc/ebnodb_8_num_beams_1_ep3_nr' \
# --model_name_or_path /home/danny911kr/joohan/seq2seq-sc2/checkpoints/seq2seq-europarl-sc/ebnodb_0_num_beams_1_ep3 \
# --ebno_db 0 \
# --nr_pretrained_path "checkpoints/seq2seq-europarl-sc/ebnodb_8_num_beams_1_ep3_nr/neural_receiver_weights" \
# --ignore_mismatched_sizes True


python train.py \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 3 \
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
    --ebno_db_min 3 \
    --ebno_db_max 10 \
    --channel_type AWGN \
    --fec_type Polar5G \
    --num_bits_per_symbol 4 \
    --overwrite_output_dir \
    --tokenizer_name facebook/bart-base \
    --pad_to_max_length \
    --dataset_config 3.0.0 \
    --bin_conv_method tanh