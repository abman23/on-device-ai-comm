eval_ebno_db="4"
metric="sbert" # bleu, sbert
testset_path='data/flickr/processed/flickr30k.json'

checkpoint_dir='checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15'
output_dir='checkpoints/on-device-ai-comm/train_CDL-A_ebnodb_5_15/CDL-A'

mkdir -p $output_dir

fec_type="Polar5G" # Polar5G, LDPC5G
fec_num_iter=20
channel_num_tx_ant="2"
channel_num_rx_ant="2"
num_bits_per_symbol="4"
EVAL_NUM_BEAMS="1"

python eval.py \
    -m "${metric}" \
    -b 8 \
    -e "${eval_ebno_db}" \
    --result-json-path "${output_dir}/flickr_${metric}_${eval_ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --prediction-json-path "${output_dir}/flickr_prediction_${eval_ebno_db}dB_${fec_type}_${channel_num_tx_ant}_${channel_num_rx_ant}_${num_bits_per_symbol}.json" \
    --fec-type "${fec_type}" \
    --fec-num-iter "${fec_num_iter}" \
    --channel-type "CDL" \
    --cdl-model "A" \
    --channel-num-tx-ant "${channel_num_tx_ant}" \
    --channel-num-rx-ant "${channel_num_rx_ant}" \
    --num-bits-per-symbol "${num_bits_per_symbol}" \
    --bin-conv-method "vector_quantization" \
    --embedding-dim 2 \
    --num-embeddings 1024 \
    --num-beams "${EVAL_NUM_BEAMS}" \
    --testset-path "${testset_path}" \
    $checkpoint_dir