#!/bin/bash
ebno_db="9"
NUM_BEAMS="1"
metric="bleu" # bleu, sbert
testset_path='data/flickr/processed/flickr30k.json'
checkpoint_path="checkpoints/seq2seq-europarl-sc/ebnodb_5_15_ep2_Polar5G_cdlA_fulltest"
output_path="checkpoints/seq2seq-europarl-sc/ebnodb_5_15_ep2_Polar5G_cdlA_fulltest/umi"

fec_type="Polar5G" # Polar5G, LDPC5G
scenario="umi" # "umi", "uma", "rma"
perfect_csi=True # True, False
channel_num_tx_ant="2"
channel_num_rx_ant="2"
num_bits_per_symbol="4"

python eval.py \
    --batch 4 \
    --metric "${metric}" \
    --ebno-db "${ebno_db}" \
    --result-json-path "${output_path}/flickr_${metric}_${ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --prediction-json-path "${output_path}/flikr_prediction_ebno_${ebno_db}_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --channel-type '3GPP-38.901' \
    --scenario "${scenario}" \
    --perfect-csi "${perfect_csi}" \
    --channel-num-tx-ant "${channel_num_tx_ant}" \
    --channel-num-rx-ant "${channel_num_rx_ant}" \
    --num-bits-per-symbol "${num_bits_per_symbol}" \
    --bin-conv-method tanh \
    --num-beams "${NUM_BEAMS}" \
    --testset-path "${testset_path}" \
    $checkpoint_path