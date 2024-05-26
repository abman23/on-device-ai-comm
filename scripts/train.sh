output_dir='checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15'
trainset_path='data/europarl/processed/train.csv'
devset_path='data/europarl/processed/test.csv'

mkdir -p $output_dir

python train.py \
    --model_name_or_path facebook/bart-base \
    --config_name facebook/bart-base \
    --tokenizer_name facebook/bart-base \
    --train_file "$trainset_path" \
    --validation_file "$devset_path" \
    --test_file "$devset_path" \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size  4 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --no_use_fast_tokenizer \
    --num_beams 1 \
    --pad_to_max_length \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --output_dir $output_dir \
    --ebno_db_min 5 \
    --ebno_db_max 15 \
    --channel_type "CDL" \
    --fec_type "Polar5G" \
    --fec_num_iter 20 \
    --cdl_model "A" \
    --channel_num_tx_ant "2" \
    --channel_num_rx_ant "2" \
    --num_bits_per_symbol "4" \
    --bin_conv_method "vector_quantization" \
    --embedding_dim 2 \
    --num_embeddings 1024 \
    --dataset_config 3.0.0