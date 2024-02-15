#!/bin/bash
ebno_db="5"
NUM_BEAMS="1"
metric="bleu" # bleu, sbert
testset_path='data/flickr/processed/flickr30k.json'
checkpoint_path="checkpoints/seq2seq-europarl-sc/ebnodb_3_10_ep3_awgn_vqfulltest2"

fec_type='Polar5G' # Polar5G, LDPC5G
num_bits_per_symbol="4"

python eval.py \
    --batch 4 \
    --metric "${metric}" \
    --ebno-db "${ebno_db}" \
    --result-json-path "${checkpoint_path}/flickr_${metric}_${ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --prediction-json-path "${checkpoint_path}/flickr_prediction_${ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --channel-type AWGN \
    --fec-type "${fec_type}" \
    --num-bits-per-symbol "${num_bits_per_symbol}" \
    --bin-conv-method vector_quantization \
    --embedding-dim 2 \
    --num-embeddings 1024 \
    --num-beams "${NUM_BEAMS}" \
    --testset-path "${testset_path}" \
    $checkpoint_path