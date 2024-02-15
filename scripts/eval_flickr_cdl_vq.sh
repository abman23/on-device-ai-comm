#!/bin/bash
ebno_db="3"
NUM_BEAMS="1"
metric="bleu" # bleu, sbert
testset_path='data/flickr/processed/flickr30k.json'
checkpoint_path="checkpoints/seq2seq-europarl-sc/ebnodb_5_15_ep3_Polar5G_cdlA_2_2_4_VQ_2_1024"

fec_type='Polar5G' # Polar5G, LDPC5G
cdl_model="A" # A,B,C,D,E
channel_num_tx_ant="2"
channel_num_rx_ant="2"
num_bits_per_symbol="4"

python eval.py \
    --batch 4 \
    --metric "${metric}" \
    --ebno-db "${ebno_db}" \
    --result-json-path "${checkpoint_path}/flickr_${metric}_${ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --prediction-json-path "${checkpoint_path}/flikr_prediction_ebno_${ebno_db}_numbeams_${NUM_BEAMS}.json" \
    --channel-type CDL \
    --cdl-model "${cdl_model}" \
    --fec-type "${fec_type}" \
    --channel-num-tx-ant "${channel_num_tx_ant}" \
    --channel-num-rx-ant "${channel_num_rx_ant}" \
    --num-bits-per-symbol "${num_bits_per_symbol}" \
    --bin-conv-method vector_quantization \
    --embedding-dim 2 \
    --num-embeddings 1024 \
    --num-beams "${NUM_BEAMS}" \
    --testset-path "${testset_path}" \
    $checkpoint_path